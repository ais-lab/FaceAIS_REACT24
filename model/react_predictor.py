import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import (GPT, MLP, Block, CrossAttention, LayerNorm,
                             SelfAttention)

SPECIAL_TOKENS = ["<PAD>", "<MASK>"]


@dataclass
class ReactPredictorConfig:
    vocab_size: int = None
    block_size: int = None

    vocab_sound_size: int = None
    sound_factor: int = 8
    vocab_face_size: int = None

    patch_size: int = 32

    is_twisted: bool = True

    n_embd: int = 512
    n_layer: int = 8
    n_head: int = 8
    dropout: float = 0.1  # for pretraining 0 is good, for finetuning try 0.1+
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    embd_pdrop: float = 0.0
    past_mask_p: float = 0.1
    bias: bool = False
    quantize_type: str = "vq"
    quantize_codebook_size: int = 512
    quantize_levels: list = field(default_factory=lambda: [8, 5, 5, 5])
    quantize_decay: float = 0.8
    quantize_commitment_weight: float = 1.0
    quantize_entropy_loss_weight: float = 0.1
    quantize_diversity_gamma: float = 1.0
    quantize_weight: float = 1.0

    # predictor mode
    predictor_mode: str = "autoregressive"  # "autoregressive" or "mask"

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

    mask_value: int = SPECIAL_TOKENS.index("<MASK>")
    padding_value: int = SPECIAL_TOKENS.index("<PAD>")
    output_dir: str = "output"
    cache: str = "cache"

    sampling_temperature: float = 0.75
    sampling_top_k: int = None
    test_extend_factor: int = 10  # this to evaluate the model's diversity
    render: bool = False
    only_speaker: bool = False
    collect_metrics_in: str = "test"  # "test" or "val"
    render_skip_step: int = 1

    use_wav2vec2_feature: bool = False

    dataconfig = None  # if type is not defined, then it would not be included when initializing the object
    quantizeconfig = None


class CrossModalBlock(nn.Module):
    # https://arxiv.org/pdf/2206.07643.pdf

    def __init__(self, config, is_last=False):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.attn_cross = CrossAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.gated_alpha = nn.Parameter(
            torch.tensor(0.0)
        )  # gated alpha is important here
        self.is_last = is_last

    def forward(self, x, y, mask_x=None, mask_y=None):
        x_hat = self.attn(self.ln_1(x), mask=mask_x)
        x = (
            x
            + x_hat
            + self.gated_alpha
            * self.attn_cross(self.ln_1(x_hat), self.ln_3(y), mask_x, mask_y)
        )
        x = x + self.mlp(self.ln_2(x))
        if self.is_last:
            return x
        return x, x


class ReactPredictor(GPT):
    def __init__(self, config: ReactPredictorConfig):
        super().__init__(config)

        assert config.vocab_size is not None  # listener face
        assert config.block_size is not None  # this would be the context length
        assert config.vocab_sound_size is not None  # speaker audio
        assert config.vocab_face_size is not None  # speaker face

        self.sound_factor = config.sound_factor
        self.wav2vec2_feature = config.use_wav2vec2_feature
        if self.wav2vec2_feature:
            self.sound_factor = (
                2  # wav2vec2 feature is 2 times longer than the face token
            )

        self.config = config

        self.config.vocab_size = config.vocab_size + len(SPECIAL_TOKENS)
        self.config.vocab_sound_size = config.vocab_sound_size + len(SPECIAL_TOKENS)
        self.config.vocab_face_size = config.vocab_face_size + len(SPECIAL_TOKENS)

        # speaker cross-modal encoder
        self.sp_cm_transformer = nn.ModuleDict(
            dict(
                sound_pool=nn.MaxPool1d(
                    self.sound_factor
                ),  # why using pool instead of projection?
                wte_sound=nn.Embedding(
                    config.vocab_sound_size,
                    config.n_embd,
                    padding_idx=config.padding_value,
                ),
                wpe_sound=nn.Embedding(
                    config.block_size * self.sound_factor,
                    config.n_embd,
                    padding_idx=config.padding_value,
                ),  # sound block size is 8 times longer
                wte_face=nn.Embedding(
                    config.vocab_face_size,
                    config.n_embd,
                    padding_idx=config.padding_value,
                ),
                wpw_face=nn.Embedding(
                    config.block_size, config.n_embd, padding_idx=config.padding_value
                ),
                drop=nn.Dropout(config.dropout),
                h_sound=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                h_face=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                h=nn.ModuleList(
                    [CrossModalBlock(config) for _ in range(config.n_layer)]
                ),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        if self.wav2vec2_feature:
            self.wav2vec2_c_proj = nn.Linear(512, config.n_embd)

        # listener cross-modal decoder
        self.lt_cm_transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.vocab_size, config.n_embd, padding_idx=config.padding_value
                ),
                wpe=nn.Embedding(
                    config.block_size, config.n_embd, padding_idx=config.padding_value
                ),
                drop=nn.Dropout(config.dropout),
                h_speaker=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                h_listener=nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),
                h=nn.ModuleList(
                    [CrossModalBlock(config) for _ in range(config.n_layer)]
                ),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )
            if "gated_alpha" in pn:
                torch.nn.init.zeros_(p)

    def forward(
        self,
        sp_sound_idx,
        sp_face_idx,
        lt_face_shifted_idx,
        mask_sp_sound=None,
        mask_sp_face=None,
        mask_lt_face=None,
        targets=None,
    ):
        device = sp_sound_idx.device
        is_twisted = self.config.is_twisted

        # # # # #
        # SPEAKER CROSS-MODAL ENCODER
        # # # # #

        if not self.wav2vec2_feature:
            # idx to emb
            sp_s_emb = self.sp_cm_transformer.wte_sound(sp_sound_idx)

        else:
            # feature to emb
            sp_s_emb = self.wav2vec2_c_proj(sp_sound_idx)

        sp_s_pos = torch.arange(0, sp_sound_idx.shape[1], device=device)
        sp_s_pos_emb = self.sp_cm_transformer.wpe_sound(sp_s_pos)
        sp_s = self.sp_cm_transformer.drop(sp_s_emb + sp_s_pos_emb)
        # max-pooling the sound token to sync with the face token
        sp_s = self.sp_cm_transformer.sound_pool(sp_s.transpose(1, 2)).transpose(1, 2)

        sp_s_mask = None
        if mask_sp_sound is not None:
            if self.wav2vec2_feature:
                mask_sp_sound = torch.min(mask_sp_sound.float(), dim=-1)[0]
            sp_s_mask = (
                self.sp_cm_transformer.sound_pool(mask_sp_sound.float()) > 0
            )  # cast boolean to float, then downsample, then cast back to boolean by using > 0
            # max-pooling the mask is just an OR operation in a window

        sp_f_emb = self.sp_cm_transformer.wte_face(sp_face_idx)
        sp_f_pos = torch.arange(0, sp_face_idx.shape[1], device=device)
        sp_f_pos_emb = self.sp_cm_transformer.wpw_face(sp_f_pos)
        sp_f = self.sp_cm_transformer.drop(sp_f_emb + sp_f_pos_emb)

        for block in self.sp_cm_transformer.h_sound:
            sp_s = block(sp_s, mask=sp_s_mask)

        for block in self.sp_cm_transformer.h_face:
            sp_f = block(sp_f, mask=mask_sp_face)

        for block in self.sp_cm_transformer.h:
            if is_twisted:
                sp_f, sp_s = block(
                    sp_s, sp_f, sp_s_mask, mask_sp_face
                )  # twisted output,
            else:
                sp_f, sp_s = block(sp_f, sp_s, mask_sp_face, sp_s_mask)  # normal output

        sp_f = self.sp_cm_transformer.ln_f(sp_f)

        # # # # #
        # LISTENER CROSS-MODAL DECODER
        # # # # #

        lt_f_shifted_emb = self.lt_cm_transformer.wte(lt_face_shifted_idx)
        lt_f_shifted_pos = torch.arange(0, lt_face_shifted_idx.shape[1], device=device)
        lt_f_shifted_pos_emb = self.lt_cm_transformer.wpe(lt_f_shifted_pos)
        lt_f_shifted = self.lt_cm_transformer.drop(
            lt_f_shifted_emb + lt_f_shifted_pos_emb
        )

        for block in self.lt_cm_transformer.h_listener:
            lt_f_shifted = block(lt_f_shifted, mask=mask_lt_face)

        for block in self.lt_cm_transformer.h_speaker:
            sp_s = block(sp_s, mask=sp_s_mask)

        for block in self.lt_cm_transformer.h:
            if is_twisted:
                sp_s, lt_f_shifted = block(
                    lt_f_shifted, sp_s, mask_lt_face, mask_sp_face
                )  # twisted output
            else:
                lt_f_shifted, sp_s = block(
                    lt_f_shifted, sp_s, mask_lt_face, mask_sp_face
                )

        lt_f_shifted = self.lt_cm_transformer.ln_f(lt_f_shifted)

        # # # # #
        # LM HEAD
        # # # # #

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(lt_f_shifted)
            loss = F.cross_entropy(
                input=logits.reshape(-1, logits.size(-1)),
                target=targets.reshape(-1).long(),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(lt_f_shifted[:, -self.config.patch_size :, :])
            loss = None

        return logits, loss
