import math
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import GPT, Block, LayerNorm
# from vq_reaction.nanoGPT import GPT, Block, LayerNorm
from model.quantizer import FSQ, LFQ
from model.quantizer import VectorQuantize as VQ


@dataclass
class FaceTokenizerConfig:
    input_dim: int = None
    emotion_output_dim: int = None
    output_dim: int = None
    block_size: int = None
    n_embd: int = 2**8
    n_layer: int = 8
    n_head: int = 8
    dropout: float = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    embd_pdrop: float = 0.0
    bias: bool = False
    quantize_type: str = "vq"
    quantize_codebook_size: int = 512
    quantize_levels: list = field(default_factory=lambda: [8, 6, 5])
    quantize_decay: float = 0.8
    quantize_commitment_weight: float = 1.0
    quantize_entropy_loss_weight: float = 0.1
    quantize_diversity_gamma: float = 1.0
    quantize_weight: float = 1.0

    # learning config
    learning_rate: float = 6e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 1000
    lr_decay_iters: int = 10000  # should be ~= max_iters per Chinchilla
    min_lr: float = (
        6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    )

    exp_weight: float = 1.0
    rot_weight: float = 1.0
    trans_weight: float = 1.0

    phase: int = 1  # 1, 2: 1st phase is to train encoder and 3dmm decoder, 2nd phase is to train emotion decoder

    output_dir: str = ""
    run_name: str = ""
    data_config = None

    checkpoint_path: str = None

    # def __init__(self, **kwargs):
    #     for k, v in kwargs.items():
    #         setattr(self, k, v)

    #     if self.quantize_type == 'lfq':
    #         self.quantize_codebook_size = 2**16
    #         self.n_embd = 16


class Quantize(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.quantize = None
        self.config = config

        if config.quantize_type == "vq":
            self.quantize = VQ(
                dim=config.n_embd,
                codebook_size=config.quantize_codebook_size,
                decay=config.quantize_decay,
                commitment_weight=config.quantize_commitment_weight,
                kmeans_init=True,
                # threshold_ema_dead_code=2,
                # stochastic_sample_codes = True,
                # sample_codebook_temp = 0.05,
            )
        elif config.quantize_type == "fsq":
            self.quantize = FSQ(
                levels=config.quantize_levels,
                dim=config.n_embd,
            )
        elif config.quantize_type == "lfq":
            self.quantize = LFQ(
                codebook_size=config.quantize_codebook_size,
                dim=config.n_embd,
            )
        else:
            raise ValueError(f"quantize type {config.quantize_type} not supported")

    def forward(self, x):
        if self.quantize is None:
            raise ValueError("quantize is None")

        if self.config.quantize_type == "vq":
            x_quantized, indices, commit_loss = self.quantize(x)

        elif self.config.quantize_type == "fsq":
            x_quantized, indices = self.quantize(x)
            commit_loss = None

        elif self.config.quantize_type == "lfq":
            x_quantized, indices, commit_loss = self.quantize(x, inv_temperature=100)

        else:
            raise ValueError(f"quantize type {self.config.quantize_type} not supported")

        return x_quantized, indices, commit_loss

    def indices_to_code(self, indices):
        if self.quantize is None:
            raise ValueError("quantize is None")

        if self.config.quantize_type == "vq":
            x_quantized = self.quantize.get_codes_from_indices(indices)

        elif self.config.quantize_type == "fsq":
            x_quantized = self.quantize.indices_to_codes(indices)

        elif self.config.quantize_type == "lfq":
            x_quantized = self.quantize.indices_to_codes(indices)

        else:
            raise ValueError(f"quantize type {self.config.quantize_type} not supported")

        return x_quantized


class FaceTokenizerTransformer(GPT):
    def __init__(self, config: FaceTokenizerConfig):
        super().__init__(config)
        self.config = config

        assert config.block_size is not None

        assert config.quantize_type in ["vq", "fsq", "lfq"]

        self.encoder = nn.ModuleDict(
            dict(
                wfe=nn.Linear(
                    config.input_dim, config.n_embd
                ),  # word feature embedding
                wpe=nn.Embedding(
                    config.block_size, config.n_embd
                ),  # position embedding
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        self.quantize = Quantize(config)

        self.decoder_to_3dmm = nn.ModuleDict(
            dict(
                wfe=nn.Linear(config.n_embd, config.n_embd),  # word feature embedding
                wpe=nn.Embedding(
                    config.block_size, config.n_embd
                ),  # word position embedding
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        self.decoder_to_emotion = nn.ModuleDict(
            dict(
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        self.lm_head_emotion = nn.Linear(
            config.n_embd, config.emotion_output_dim, bias=False
        )

        self.lm_head_3dmm = nn.Linear(config.n_embd, config.output_dim, bias=False)
        # no tying in the tokenizers, since output_dim != vocab_size

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.decoder.wte.weight = (
        #     self.lm_head.weight
        # )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def freeze_decoder_emotion(self):
        for param in self.decoder_to_emotion.parameters():
            param.requires_grad = False

        self.lm_head_emotion.weight.requires_grad = False

    def unfreeze_decoder_emotion(self):
        for param in self.decoder_to_emotion.parameters():
            param.requires_grad = True

        self.lm_head_emotion.weight.requires_grad = True

    def freeze_decoder_3dmm(self):
        for param in self.decoder_to_3dmm.parameters():
            param.requires_grad = False

        self.lm_head_3dmm.weight.requires_grad = False

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_quantize(self):
        for param in self.quantize.parameters():
            param.requires_grad = False

    def unfreeze_tokenizer(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_tokenizer(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, targets=None):
        b, t, c = x.size()  # batch, sequence length, channels feature
        device = x.device
        pos = self.get_pos(t, device)

        x, indices, quantize_loss = self.encode(x, pos)
        x, x_emotion = self.decode(x, pos)

        logits_3dmm = self.lm_head_3dmm(x)
        logits_emotion = self.lm_head_emotion(x_emotion)

        return logits_3dmm, logits_emotion, quantize_loss

    def encode(self, x, pos):
        tok_emb = self.encoder.wfe(x)  # shape (b, t, c)
        pos_emb = self.encoder.wpe(pos)  # shape (t, c)
        x = tok_emb + pos_emb  # shape (b, t, c)
        x = self.encoder.drop(x)  # shape (b, t, c)
        for block in self.encoder.h:
            x = block(x)
        x = self.encoder.ln_f(x)  # shape (b, t, c)

        # QUANTIZE FORWARD

        x, indices, quantize_loss = self.quantize(x)
        return x, indices, quantize_loss

    def decode(self, x, pos):
        tok_emb = self.decoder_to_3dmm.wfe(x)  # shape (b, t, c)
        pos_emb = self.decoder_to_3dmm.wpe(pos)

        x_3dmm = self.decoder_to_3dmm.drop(tok_emb + pos_emb)
        for block in self.decoder_to_3dmm.h:
            x_3dmm = block(x_3dmm)
        x_3dmm = self.decoder_to_3dmm.ln_f(x_3dmm)

        x_emotion = self.decoder_to_emotion.drop(tok_emb + pos_emb)
        for block in self.decoder_to_emotion.h:
            x_emotion = block(x_emotion)
        x_emotion = self.decoder_to_emotion.ln_f(x_emotion)

        return x_3dmm, x_emotion

    def get_pos(self, t, device):
        # assert (
        #     t <= self.config.block_size
        # ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        return pos

    def indices_to_recon(self, indices, pos):
        x = self.quantize.indices_to_code(indices)

        x = x.reshape(x.shape[0], -1, self.config.n_embd)

        output_3dmm, output_emotion = self.decode(x, pos)

        _3dmm = self.lm_head_3dmm(output_3dmm)
        _emotion = self.lm_head_emotion(output_emotion)

        return _3dmm, _emotion

    def tokenize(self, x):
        block_size = self.config.block_size

        pos = self.get_pos(block_size, x.device)
        for i in range(x.shape[1] // block_size):
            x_hat = x[:, i * block_size : (i + 1) * block_size, :]
            _, _indices, _ = self.encode(x_hat, pos)
            if i == 0:
                indices = _indices

            else:
                indices = torch.cat((indices, _indices), dim=1)

        return indices

    def get_3dmm_emotion(self, indices):
        block_size = self.config.block_size

        pos = self.get_pos(block_size, indices.device)

        for i in range(0, indices.shape[1], block_size):
            _indices = indices[:, i : i + block_size]
            _3dmm, _emotion = self.indices_to_recon(_indices, pos)

            if i == 0:
                output_3dmm = _3dmm
                output_emotion = _emotion

            else:
                output_3dmm = torch.cat((output_3dmm, _3dmm), dim=1)
                output_emotion = torch.cat((output_emotion, _emotion), dim=1)

        return output_3dmm, output_emotion


class FaceTokenizerANN(nn.Module):
    def __init__(self, config, loss_fn):
        super().__init__()
        self.config = config

        self.encoder = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.input_dim, config.n_embd),
            nn.ReLU(),
            nn.Linear(config.n_embd, config.n_embd),
            nn.ReLU(),
            nn.LayerNorm((config.block_size, config.n_embd)),
        )

        self.quantize = Quantize(config)

        self.decoder = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.n_embd, config.n_embd),
            nn.ReLU(),
            nn.Linear(config.n_embd, config.output_dim),
            nn.ReLU(),
            nn.LayerNorm((config.block_size, config.output_dim)),
        )

        self.loss_fn = loss_fn

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * self.config.output_dim),
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        x = self.encoder(x)

        _, indices, quant_loss = self.quantize(x)
        x = self.quantize.indices_to_code(indices)
        x = x.reshape(x.shape[0], -1, 256)

        x = self.decoder(x)

        loss = None
        if targets is not None:
            loss = self.loss_fn(x, targets, quant_loss)
            return x, loss

        return x, loss, quant_loss


if __name__ == "__main__":
    pass
