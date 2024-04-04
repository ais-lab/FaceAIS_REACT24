import json
import os

import torch
import torch.nn.functional as F
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from basetrainer.basetrainer import BaseTrainer
from dataset.dataloader import Dataconfig, ReactDataModule
from model.face_tokenizer import FaceTokenizerConfig
from model.react_predictor import ReactPredictor, ReactPredictorConfig
from model.wav2vec2_feature_extractor import Wav2Vec2ForFeatureExtraction
from model.wav2vec_tokenizer import SoundTokenizer
from p1_face_tokenizer import VQPretrainer
from render import Render

os.environ["PYTHONHASHSEED"] = str(0)


def pad(seq, max_lenth, padding_value=0, side="right", has_features=False):
    """
    Pad a sequence with a given maximum length.

    Args:
        seq (torch.Tensor): The input sequence to be padded.
        max_length (int): The maximum length of the padded sequence.
        padding_value (int, optional): The value used for padding. Defaults to 0.

    Returns:
        torch.Tensor: The padded sequence.
    """
    time_dim = 1
    if has_features:
        seq = seq.transpose(1, 2)
        time_dim = 2
    if seq.size(time_dim) != max_lenth:
        if side == "right":
            seq = F.pad(
                seq,
                (0, max_lenth - seq.size(time_dim)),
                mode="constant",
                value=padding_value,
            )
        else:
            seq = F.pad(
                seq,
                (max_lenth - seq.size(time_dim), 0),
                mode="constant",
                value=padding_value,
            )
    if has_features:
        seq = seq.transpose(1, 2)

    mask = seq != padding_value

    return seq, mask


class VQPredictor(BaseTrainer):
    def __init__(
        self,
        config: ReactPredictorConfig,
        facial_tokenizer: VQPretrainer = None,
        sound_tokenizer: SoundTokenizer = None,
        sound_feature_extractor: Wav2Vec2ForFeatureExtraction = None,
        render: Render = None,
        *args,
        **kwargs,
    ):
        super().__init__(config, *args, **kwargs)

        self.model = ReactPredictor(config)

        self.face_tokenizer = facial_tokenizer

        self.sound_tokenizer = sound_tokenizer

        self.sound_feature_extractor = sound_feature_extractor

        if facial_tokenizer is not None:
            # set tokenizer to eval mode and freeze
            self.face_tokenizer.eval()
            self.face_tokenizer.freeze()

        os.makedirs(config.output_dir, exist_ok=True)

        self.config = config

        self.render = render

    def _common_step(self, batch, batch_idx):
        """
        Perform a common step in the training or prediction process.

        Args:
            batch (tuple): A tuple containing the input data for the step.
            batch_idx (int): The index of the current batch.

        Returns:
            tuple: A tuple containing the loss, logits, and the original batch.
        """
        # sp is speaker
        # lt is listener

        max_mask = self.config.block_size
        past_mask_p = self.config.past_mask_p
        patch_size = self.config.patch_size

        (
            _,
            sp_audio_clip,
            sp_emotion,
            sp_3dmm,
            _,
            lt_audio_clip,
            lt_emotion,
            lt_3dmm,
            _,
        ) = batch

        sp_face_tok = self.face_tokenizer.model.tokenize(
            sp_3dmm[:, : self.config.block_size, :]
        )
        sp_face_tok = (
            sp_face_tok + 2
        )  # add 2 to avoid 0 and 1, which are used for padding and mask

        if not self.config.use_wav2vec2_feature:
            sp_audio_tok = self.audio_to_token(sp_audio_clip, sp_face_tok.shape[1])

            sp_audio_tok = (
                sp_audio_tok + 2
            )  # add 2 to avoid 0 and 1, which are used for padding and mask
        else:
            sp_audio_tok = self.audio_to_feature(sp_audio_clip, sp_face_tok.shape[1])

        # each sound token is 5ms, or 200 tokens is 1s
        # each face token is 40ms or 25 tokens is 1s
        # so we need to downsample the sound token to 8 times to match the face token

        sp_audio_tok, sp_audio_pad_mask = pad(
            sp_audio_tok,
            padding_value=self.config.padding_value,
            max_lenth=self.config.block_size * self.config.sound_factor,
            side="left",
            has_features=self.config.use_wav2vec2_feature,
        )

        lt_face_tok = self.face_tokenizer.model.tokenize(lt_3dmm)

        lt_face_tok = (
            lt_face_tok + 2
        )  # add 2 to avoid 0 and 1, which are used for padding and mask

        lt_past_tok = lt_face_tok[:, : self.config.block_size].clone()

        targets = lt_face_tok[
            :, patch_size : self.config.block_size + patch_size
        ].clone()  # shift the target to the right by patch_size token, the same as gpt

        # random mask the past token
        face_mask = None
        sound_mask = None

        # posibility to mask the past token
        m = torch.bernoulli(torch.tensor([past_mask_p]))
        if m > 0:
            for i in range(lt_past_tok.size(0)):
                mask_index = torch.randint(0, max_mask, (1,)).item()
                _face_mask = torch.zeros_like(lt_past_tok[i])
                _face_mask[-mask_index:] = 1
                _sound_mask = torch.zeros_like(sp_audio_tok[i])
                _sound_mask[-mask_index * self.config.sound_factor :] = 1
                if face_mask is None:
                    face_mask = _face_mask.unsqueeze(0)
                    sound_mask = _sound_mask.unsqueeze(0)
                else:
                    face_mask = torch.cat((face_mask, _face_mask.unsqueeze(0)), dim=0)
                    sound_mask = torch.cat(
                        (sound_mask, _sound_mask.unsqueeze(0)), dim=0
                    )
            face_mask = face_mask > 0
            sound_mask = sound_mask > 0

        logits, loss = self.model(
            sp_sound_idx=sp_audio_tok,
            sp_face_idx=sp_face_tok,
            lt_face_shifted_idx=lt_past_tok,
            mask_sp_sound=sound_mask if sound_mask is not None else sp_audio_pad_mask,
            mask_sp_face=face_mask,
            mask_lt_face=face_mask,
            targets=targets,
        )

        return loss, logits, batch

    def training_step(self, batch, batch_idx):
        loss, logits, batch = self._common_step(batch, batch_idx)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, batch_size=logits.size(0)
        )

        if self.global_step < self.config.warmup_iters:
            _, scheduler = self.lr_schedulers()
            scheduler.step()

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, batch = self._common_step(batch, batch_idx)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, batch_size=logits.size(0)
        )

        return loss

    def audio_to_token(self, sp_audio_clip, num_frame):
        audio_list = []
        for idx, audio_path in enumerate(sp_audio_clip[0]):
            start_frame = float(sp_audio_clip[1][idx])
            sp_audio = self.sound_tokenizer.read_audio(
                audio_path,
                start_frame=start_frame,
                fps=25,
                num_frame=num_frame,
                device=self.device,
            )
            audio_list.append(sp_audio)
        sp_audio = torch.stack(audio_list)

        sp_audio_tok = self.sound_tokenizer(
            sp_audio
        )  # produce 2 group of token of the same sound, each group is 10ms per token, so 2 groups would be 5ms per token
        sp_audio_tok = sp_audio_tok.reshape(
            sp_audio_tok.shape[0], -1
        )  # flatten 2 group of tokens to 1 group

        return sp_audio_tok

    def audio_to_feature(self, sp_audio_clip, num_frame):
        audio_list = []
        for idx, audio_path in enumerate(sp_audio_clip[0]):
            start_frame = float(sp_audio_clip[1][idx])
            sp_audio = self.sound_feature_extractor(
                audio_path,
                start_frame=start_frame,
                fps=25,
                num_frame=num_frame,
                device=self.device,
            )
            audio_list.append(sp_audio)
        sp_audio = torch.stack(audio_list, dim=0).squeeze()

        return sp_audio

    def on_test_start(self) -> None:
        super().on_test_start()
        self.listener_pred_emotion = None
        self.listener_gt = None
        self.speaker_gt = None
        self.listener_pred_3dmm = None

        self.metrics = None

    def test_step(
        self,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if batch_idx % self.config.render_skip_step != 0 and self.config.render:
            return
        render = self.config.render
        temperature = self.config.sampling_temperature
        top_k = self.config.sampling_top_k
        input_lag = (
            0  # just like human, the model should express reaction after some time.
        )

        (
            sp_video_clip,
            sp_audio_clip,
            sp_emotion,
            sp_3dmm,
            lt_video_clip,
            lt_audio_clip,
            lt_emotion,
            lt_3dmm,
            lt_ref_image,
        ) = batch

        # instead of loading the sp_video_clip and lt_video_clip
        # we take the address of the frame in the original video
        # and provide it to the render to render the video
        # should not load the video here, because it will take a lot of memory

        batch_size = sp_3dmm.size(0)
        patch_size = self.config.patch_size

        sp_face_tok = self.face_tokenizer.model.tokenize(sp_3dmm)
        sp_face_tok = (
            sp_face_tok + 2
        )  # add 2 to avoid 0 and 1, which are used for padding and mask

        if not self.config.use_wav2vec2_feature:
            sp_audio_tok = self.audio_to_token(sp_audio_clip, sp_face_tok.shape[1])

            sp_audio_tok = (
                sp_audio_tok + 2
            )  # add 2 to avoid 0 and 1, which are used for padding and mask
        else:
            sp_audio_tok = self.audio_to_feature(sp_audio_clip, sp_face_tok.shape[1])
            if batch_size == 1:
                sp_audio_tok = sp_audio_tok.unsqueeze(0)
        # each sound token is 5ms, or 200 tokens is 1s
        # each face token is 40ms or 25 tokens is 1s
        # so we need to downsample the sound token to 8 times to match the face token

        sp_audio_tok, sp_audio_pad_mask = pad(
            sp_audio_tok,
            padding_value=self.config.padding_value,
            max_lenth=sp_face_tok.size(1) * self.config.sound_factor,
            side="left",
            has_features=self.config.use_wav2vec2_feature,
        )

        # predict the next token
        # sample from the distribution
        # append it to the ouput sequence
        # repeat until the end of the sequence
        shifted_lt = torch.zeros_like(sp_face_tok).to(
            self.device
        )  # zero is the padding token
        for i in range(0, sp_face_tok.size(1), patch_size):
            start = max(0, i + 1 - input_lag - self.config.block_size)
            end = min(i + 1 - input_lag, sp_face_tok.shape[1])

            sp_sound_idx_cond = (
                sp_audio_tok[
                    :, start * self.config.sound_factor : end * self.config.sound_factor
                ]
                .clone()
                .detach()
            )
            sp_face_idx_cond = sp_face_tok[:, start:end].clone().detach()
            lt_face_idx_cond = shifted_lt[:, -self.config.block_size :].detach()

            if end <= 0:
                if self.config.use_wav2vec2_feature:
                    sp_sound_idx_cond = torch.zeros(
                        (
                            sp_audio_tok.shape[0],
                            self.config.block_size * self.config.sound_factor,
                            512,
                        )  # 512 is the feature size of wav2vec2
                    ).to(self.device)
                else:
                    sp_sound_idx_cond = (
                        torch.zeros(
                            (
                                sp_audio_tok.shape[0],
                                self.config.block_size * self.config.sound_factor,
                            )
                        )
                        .long()
                        .to(self.device)
                    )
                sp_face_idx_cond = (
                    torch.zeros((sp_face_idx_cond.shape[0], self.config.block_size))
                    .long()
                    .to(self.device)
                )

            sp_sound_idx_cond, sp_audio_tok_masked = pad(
                sp_sound_idx_cond,
                padding_value=self.config.padding_value,
                max_lenth=self.config.block_size * self.config.sound_factor,
                side="left",
                has_features=self.config.use_wav2vec2_feature,
            )

            sp_face_idx_cond, sp_face_tok_masked = pad(
                sp_face_idx_cond,
                padding_value=self.config.padding_value,
                max_lenth=self.config.block_size,
                side="left",
            )

            lt_face_idx_cond, lt_face_idx_cond_mask = pad(
                lt_face_idx_cond,
                padding_value=self.config.padding_value,
                max_lenth=self.config.block_size,
                side="left",
            )

            logits, _ = self.model(
                sp_sound_idx=sp_sound_idx_cond,
                sp_face_idx=sp_face_idx_cond,
                lt_face_shifted_idx=lt_face_idx_cond,
                mask_sp_sound=sp_audio_tok_masked,
                mask_lt_face=lt_face_idx_cond_mask,
                mask_sp_face=sp_face_tok_masked,
            )
            sampled_idx = self.sampling_step(logits, temperature, top_k)

            # add the sampled token to the shifted_lt

            shifted_lt = shifted_lt[:, self.config.patch_size :]

            shifted_lt = torch.cat([shifted_lt, sampled_idx], dim=1)

            # print(sampled_idx, i, start, end)
        shifted_lt = (
            shifted_lt - 2
        )  # remove the 2 we added to avoid 0 and 1 in training
        decode_3dmm, decode_emotion = self.face_tokenizer.model.get_3dmm_emotion(
            shifted_lt
        )

        # colect statistics data
        if self.listener_pred_emotion is None:
            self.listener_pred_emotion = decode_emotion.cpu()
            self.listener_pred_3dmm = decode_3dmm.cpu()
            self.listener_gt = lt_emotion.cpu()
            self.speaker_gt = sp_emotion.cpu()

        else:
            self.listener_pred_emotion = torch.cat(
                [self.listener_pred_emotion, decode_emotion.cpu()], dim=0
            )

            self.listener_pred_3dmm = torch.cat(
                [self.listener_pred_3dmm, decode_3dmm.cpu()], dim=0
            )

            if dataloader_idx == 0:
                self.listener_gt = torch.cat(
                    [self.listener_gt, lt_emotion.cpu()], dim=0
                )
                self.speaker_gt = torch.cat([self.speaker_gt, sp_emotion.cpu()], dim=0)

        # render video
        if dataloader_idx == 0 and render:
            batch_size = decode_3dmm.size(0)
            for bs in range(batch_size):
                mesh_dir_path = os.path.join(
                    self.config.output_dir,
                    "mesh_video_b{}_ind{}".format(str(batch_idx + 1), str(bs + 1)),
                )
                fake_dir_path = os.path.join(
                    self.config.output_dir,
                    "fake_video_b{}_ind{}".format(str(batch_idx + 1), str(bs + 1)),
                )

                for t in range(decode_3dmm.size(1)):
                    self.render.single_frame_render_mesh(
                        path=mesh_dir_path,
                        name=f"frame_{t}",
                        facial_3dmm_vector=decode_3dmm[bs][t].unsqueeze(0),
                    )
                    self.render.single_frame_render_fake(
                        path=fake_dir_path,
                        name=f"frame_{t}",
                        facial_3dmm_vector=decode_3dmm[bs][t],
                        reference_img=lt_ref_image[bs],
                        is_final=True if t == decode_3dmm.size(1) - 1 else False,
                    )

                # convert the frames to video
                # os.system(
                #     f"ffmpeg -r 30 -i {mesh_dir_path}/frame_%d.png -vcodec mpeg4 -y {mesh_dir_path}.mp4"
                # )
                # os.system(
                #     f"ffmpeg -r 30 -i {fake_dir_path}/frame_%d.png -vcodec mpeg4 -y {fake_dir_path}.mp4"
                # )

                lt_gt_frame_address = [address[bs] for address in lt_video_clip]
                sp_gt_frame_address = [address[bs] for address in sp_video_clip]

                # write json file for combine video later
                with open(
                    os.path.join(
                        self.config.output_dir,
                        "b{}_ind{}.json".format(str(batch_idx + 1), str(bs + 1)),
                    ),
                    "w",
                ) as f:
                    json.dump(
                        {
                            "lt_gt_frame_address": lt_gt_frame_address,
                            "sp_gt_frame_address": sp_gt_frame_address,
                            "mesh_video_address": os.path.join(
                                self.config.output_dir,
                                "mesh_video_b{}_ind{}".format(
                                    str(batch_idx + 1), str(bs + 1)
                                ),
                            ),
                            "fake_video_address": os.path.join(
                                self.config.output_dir,
                                "fake_video_b{}_ind{}".format(
                                    str(batch_idx + 1), str(bs + 1)
                                ),
                            ),
                        },
                        f,
                    )

        if "only_submit_video" in self.config.output_dir:
            os.system(
                f"python render_final_video.py --output-dir {self.config.output_dir}"
            )

    def on_test_epoch_end(self) -> None:
        batch_size = self.speaker_gt.size(0)
        length = self.listener_pred_emotion.size(-2)
        feature_size_emotion = self.listener_pred_emotion.size(-1)
        feature_size_3dmm = self.listener_pred_3dmm.size(-1)

        self.listener_pred_emotion = self.listener_pred_emotion.reshape(
            batch_size, -1, length, feature_size_emotion
        )

        self.listener_pred_3dmm = self.listener_pred_3dmm.reshape(
            batch_size, -1, length, feature_size_3dmm
        )

        torch.save(
            self.listener_pred_emotion,
            os.path.join(
                self.config.output_dir,
                # self.config.collect_metrics_in,
                "listener_pred_emotion.pt",
            ),
        )

        torch.save(
            self.listener_pred_3dmm,
            os.path.join(
                self.config.output_dir,
                # self.config.collect_metrics_in,
                "listener_pred_3dmm.pt",
            ),
        )

        torch.save(
            self.listener_gt,
            os.path.join(
                self.config.output_dir,
                # self.config.collect_metrics_in,
                "listener_gt.pt",
            ),
        )
        torch.save(
            self.speaker_gt,
            os.path.join(
                self.config.output_dir,
                # self.config.collect_metrics_in,
                "speaker_gt.pt",
            ),
        )

        if self.config.only_speaker:
            # no groundtruth for listener, so we can't compute the metrics
            return

        # self.metrics = metrics_compute(
        #     dataset_path=self.config.dataconfig.dataset_path,
        #     listener_pred=self.listener_pred,
        #     speaker_gt=self.speaker_gt,
        #     listener_gt=self.listener_gt,
        #     # fid_dir=self.config.output_dir,
        #     device=self.device,
        #     # p=4,
        #     past_metrics=self.metrics,
        # )

        # print(
        #     f"TLCC: {self.metrics.TLCC.avg}, \
        #     FRC: {self.metrics.FRC.avg}, \
        #     FRD: {self.metrics.FRD.avg}, \
        #     FRDvs: {self.metrics.FRDvs.avg}, \
        #     FRVar: {self.metrics.FRVar.avg}, \
        #     smse: {self.metrics.smse.avg}, \
        #     FRRea: {self.metrics.FRRea.avg}"
        # )

        # self.log_dict(
        #     {
        #         "TLCC": self.metrics.TLCC.avg,
        #         "FRC": self.metrics.FRC.avg,
        #         "FRD": self.metrics.FRD.avg,
        #         "FRDvs": self.metrics.FRDvs.avg,
        #         "FRVar": self.metrics.FRVar.avg,
        #         "smse": self.metrics.smse.avg,
        #         "FRRea": self.metrics.FRRea.avg,
        #     },
        #     on_epoch=True,
        #     logger=True,
        # )

        # self.listener_gt = None
        # self.listener_pred = None
        # self.speaker_gt = None

    def sampling_step(self, logits, temperature=1.0, top_k=None, top_p=None):
        # pluck the logits at the final step and scale by desired temperature
        logits = logits / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))

            # loop through each timestep and mask out the logits below the top k
            for i in range(logits.size(1)):
                _logits = logits[:, i]
                _logits[_logits < torch.min(v[:, i])] = -float("Inf")

                if i == 0:
                    new_logits = _logits.unsqueeze(
                        0
                    )  # unsqueeze to create a time dimension
                else:
                    new_logits = torch.cat((new_logits, _logits.unsqueeze(0)), dim=0)

            # transpose to flip the time and batch dimension
            logits = new_logits.transpose(0, 1)

        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)

        # sample from the distribution
        for i in range(probs.size(1)):
            idx_next = torch.multinomial(probs[:, i], num_samples=1)
            if i == 0:
                sampled_idx = idx_next
            else:
                sampled_idx = torch.cat((sampled_idx, idx_next), dim=1)

        return sampled_idx

    def configure_optimizers(self):
        optimizer = self.model.configure_optimizers(
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            device_type=self.device,
        )

        warmup_duration = self.config.warmup_iters

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.000001,
            end_factor=1,
            total_iters=warmup_duration,
        )

        red_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=10000,
            min_lr=self.config.min_lr,
            verbose=True,
        )

        lr_scheduler = {
            "scheduler": red_plateau,
            "interval": "epoch",
            "frequency": 2,
            "monitor": "val_loss",
        }

        return (
            [optimizer],
            [lr_scheduler, {"scheduler": warmup}],
        )


def main(
    test=False,
    resume=False,
    resume_checkpoint=None,
    test_checkpoint=None,
    tokenizer_checkpoint=None,
    output_dir=None,
    dataset_path=None,
):
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    # resume = False
    # test = False
    # dataset_path = "/home/tien/playground_facereconstruction/data/react_2024"
    # submit_video_only = False
    # tokenizer_checkpoint = "/home/tien/playground_facereconstruction/output/quantize_vq_pretrain_3/epoch=185-step=37200.ckpt"
    # output_dir = "/home/tien/playground_facereconstruction/output/quantize_vq_predictor_fiber_attention_block"
    # resume_checkpoint = "/home/tien/playground_facereconstruction/output/quantize_vq_predictor_1/epoch=399-step=40000.ckpt"
    # test_checkpoint = "/home/tien/playground_facereconstruction/output/quantize_vq_predictor_fiber_attention_block/epoch=91-step=9200.ckpt"

    # # # # # # #
    # C O N F I G
    # # # # # # #

    tokenizer_config = FaceTokenizerConfig(
        input_dim=58,
        output_dim=58,
        emotion_output_dim=25,
        block_size=32,
        n_embd=252,  # if lfq then 144, else 252
        n_head=12,
        n_layer=12,
        quantize_type="fsq",  # lfq, vq, fsq
        quantize_codebook_size=2048,  # the vocab size: 512, 1024, 2048
        quantize_levels=[8, 5, 5, 5],
    )

    # follow config of https://arxiv.org/pdf/2212.05199.pdf
    predictor_config = ReactPredictorConfig(
        # ACRCHITECTURE CONFIG
        vocab_size=tokenizer_config.quantize_codebook_size,  # this is the vocab size of the face token but in predictor
        vocab_face_size=tokenizer_config.quantize_codebook_size,  # this the vocab size of the face token
        vocab_sound_size=320,  # if use wav2vec2 feature, the sound vocab size is not needed
        is_twisted=False,  # Tested: twist and no twist make little difference
        block_size=256,  # this is context length, pad if not enough, cut if too long, tried 128, 256
        n_embd=360,  # TODO: increase the size of the model
        n_head=12,
        n_layer=8,
        sound_factor=2,  # the factor to downsample the sound token to match the face token
        use_wav2vec2_feature=True,  # use wav2vec2 vector instead of vq-wav2vec make the model's output more synchronized with sound
        patch_size=32,  # should be any factor of block_size of tokenizer?
        # TRAINING CONFIG
        learning_rate=0.001,  # this learningrate produce best result
        warmup_iters=200,  # should be 2~5% of the total iteration
        min_lr=6e-4,
        past_mask_p=0.5,  # 30% of the time we mask a random number of past token
        dropout=0.1,  # NanoGPT suggest no dropout, but it seems overfit if no dropout
        attn_dropout=0.1,
        embd_pdrop=0.1,
        resid_dropout=0.1,
        bias=True,
        # TEST CONFIG
        sampling_top_k=512,
        sampling_temperature=1,  # the higher the temperature, the more random the output
        test_extend_factor=10,
        only_speaker=False,
        collect_metrics_in="val",
        render_skip_step=1,
    )

    data_config = Dataconfig(
        dataset_path=dataset_path,
        batch_size=16,
        num_workers=12,
        img_size=256,
        crop_size=224,
        clip_length=(
            predictor_config.block_size + tokenizer_config.block_size
            if not test
            else 736
        ),
        test_extend_factor=predictor_config.test_extend_factor,
        submit_video_only=False,
    )
    predictor_config.dataconfig = data_config
    tokenizer_config.checkpoint_path = tokenizer_checkpoint
    predictor_config.quantizeconfig = tokenizer_config

    # # # # # # # # # # # # # # # # # # # # # # # #
    # M O D E L   P R E P A R A T I O N
    # # # # # # # # # # # # # # # # # # # # # # # #

    face_tokenizer = VQPretrainer.load_from_checkpoint(
        checkpoint_path=tokenizer_config.checkpoint_path,
        config=tokenizer_config,
    )

    if predictor_config.use_wav2vec2_feature:
        sound_feature_extractor = Wav2Vec2ForFeatureExtraction()
        sound_tokenizer = None
        predictor_config.sound_factor = 2
    else:
        sound_tokenizer = SoundTokenizer()
        sound_feature_extractor = None
        predictor_config.sound_factor = 8

    run_name = f"{tokenizer_config.quantize_type}-wav2vec2-patch32-context256-vocab2048"
    version = f"k{predictor_config.sampling_top_k}-{predictor_config.sampling_temperature}-p{predictor_config.patch_size}-output"

    predictor_config.output_dir = os.path.join(
        output_dir,
        run_name,
        version,
    )
    os.makedirs(predictor_config.output_dir, exist_ok=True)
    render = Render("cuda")

    if resume:
        module = VQPredictor.load_from_checkpoint(
            checkpoint_path=resume_checkpoint,
            config=predictor_config,
            facial_tokenizer=face_tokenizer,
            sound_tokenizer=sound_tokenizer,
            sound_feature_extractor=sound_feature_extractor,
            render=render,
        )
    else:
        module = VQPredictor(
            config=predictor_config,
            facial_tokenizer=face_tokenizer,
            sound_tokenizer=sound_tokenizer,
            sound_feature_extractor=sound_feature_extractor,
            render=render,
        )

    # # # # # # # # # # # # # # # # # # # # # # # #
    # T R A I N E R   P R E P A R A T I O N
    # # # # # # # # # # # # # # # # # # # # # # # #

    wandb_logger = WandbLogger(
        project="quantize_vq_predictor",
        config=predictor_config,
        name=run_name,
    )
    tensorboard_logger = TensorBoardLogger(save_dir=output_dir)

    if not test:
        datamodule = ReactDataModule(
            conf=data_config,
            # load raw audio because we will use vq-wav2vec to encode audio
            # load 3dmm because we will use vq-vae to encode 3dmm
            load_3dmm=True,
            load_audio=True,
            load_emotion=True,
            load_ref=False,
            load_video=False,
            load_raw_audio=True,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=output_dir,
            save_top_k=1,
            mode="min",
            save_last=True,
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        devices=1,
        min_epochs=1,
        max_epochs=1000,
        precision=32,
        accelerator="auto",
        callbacks=(
            [
                checkpoint_callback,
                lr_monitor,
            ]
            if not test
            else None
        ),
        check_val_every_n_epoch=2,
        enable_checkpointing=True if not test else False,
        log_every_n_steps=5,
        logger=(
            [
                wandb_logger,
                tensorboard_logger,
            ]
            if not test
            else None
        ),
    )

    # # # # # # # # # # # # # # # # # # # # # # # #
    # T R A I N I N G
    # # # # # # # # # # # # # # # # # # # # # # # #

    if not test:
        trainer.fit(module, datamodule=datamodule)
        best_model_path = checkpoint_callback.best_model_path
        wandb_logger.log_hyperparams({"best_model_path": best_model_path})
        wandb_logger.log_hyperparams(
            {"last_model_path": checkpoint_callback.last_model_path}
        )
        test_checkpoint = best_model_path

    # # # # # # # # # # # # # # # # # # # # # # # #
    # T E S T I N G   A N D   R E N D E R I N G
    # # # # # # # # # # # # # # # # # # # # # # # #

    def run_test(
        is_render=True,
        submit_video_only=True,
        collect_metrics_in=predictor_config.collect_metrics_in,
        render_skip_step=predictor_config.render_skip_step,
    ):
        module.config.render = is_render
        module.config.output_dir = os.path.join(
            output_dir,
            run_name,
            version,
            collect_metrics_in,
            "render_for_fid" if not submit_video_only else "only_submit_video",
        )
        os.makedirs(module.config.output_dir, exist_ok=True)
        module.config.render_skip_step = render_skip_step
        data_config.batch_size = 8
        data_config.test_extend_factor = 1 if is_render else 10
        data_config.clip_length = 736 if is_render else 736
        data_config.submit_video_only = submit_video_only
        datamodule = ReactDataModule(
            conf=data_config,
            load_3dmm=True,
            load_audio=True,
            load_emotion=True,
            load_ref=True if is_render else False,
            load_video=True if is_render else False,
            load_raw_audio=True,
            load_video_address=True,
            only_speaker=predictor_config.only_speaker,
            collect_metrics_in=collect_metrics_in,
        )
        trainer.test(module, datamodule=datamodule, ckpt_path=test_checkpoint)

    print("RENDERING THE REQUIRED VIDEOS")
    run_test(is_render=True, submit_video_only=True, collect_metrics_in="val")

    print("RENDERING FOR FID")
    run_test(
        is_render=True,
        submit_video_only=False,
        collect_metrics_in="val",
        render_skip_step=50,
    )
    run_test(
        is_render=True,
        submit_video_only=False,
        collect_metrics_in="test",
        render_skip_step=50,
    )

    print("COLLECTING PREDICTION IN VAL")
    run_test(is_render=False, submit_video_only=False, collect_metrics_in="val")

    print("COLLECTING PREDICTION IN TEST")
    run_test(is_render=False, submit_video_only=False, collect_metrics_in="test")

    # Due to a multiprocessing error in evaluation method, we have to run it in another process.

    print("COMPUTING METRICS FOR VAL")
    val_dir = os.path.join(
        output_dir,
        run_name,
        version,
        "val",
        "render_for_fid",
    )
    os.system(
        f"python compute_metrics.py --output_dir {val_dir} --dataset_path {dataset_path} --val_test val"
    )

    print("COMPUTING METRICS FOR TEST")
    test_dir = os.path.join(
        output_dir,
        run_name,
        version,
        "test",
        "render_for_fid",
    )
    os.system(
        f"python compute_metrics.py --output_dir {test_dir} --dataset_path {dataset_path} --val_test test"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--resume", action="store_true", help="Resume the training")
    parser.add_argument(
        "--resume_checkpoint", type=str, help="The checkpoint to resume the training"
    )
    parser.add_argument(
        "--test_checkpoint", type=str, help="The checkpoint to test the model"
    )
    parser.add_argument(
        "--tokenizer_checkpoint", type=str, help="The checkpoint of the tokenizer"
    )
    parser.add_argument("--output_dir", type=str, help="The output directory")
    parser.add_argument("--dataset_path", type=str, help="The path to the dataset")

    args = parser.parse_args()
    main(
        test=args.test,
        resume=args.resume,
        resume_checkpoint=args.resume_checkpoint,
        test_checkpoint=args.test_checkpoint,
        tokenizer_checkpoint=args.tokenizer_checkpoint,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
    )

    # sample command
    # python p2_react_predictor.py --test --test_checkpoint /home/tien/playground_facereconstruction/output/quantize_vq_predictor_fiber_attention_block/epoch=91-step=9200.ckpt --tokenizer_checkpoint /home/tien/playground_facereconstruction/output/quantize_vq_pretrain_3/epoch=185-step=37200.ckpt --output_dir /home/tien/playground_facereconstruction/output/quantize_vq_predictor_fiber_attention_block --dataset_path /home/tien/playground_facereconstruction/data/react_2024
