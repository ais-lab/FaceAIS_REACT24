import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from basetrainer.basetrainer import BaseTrainer
from dataset.dataloader import Dataconfig, ReactDataModule
from model.face_tokenizer import (FaceTokenizerANN, FaceTokenizerConfig,
                                  FaceTokenizerTransformer)
from render import Render


def calc_vq_loss(
    pred,
    target,
    quant_loss,
    quant_loss_weight=1.0,
    LLoss=True,
    BlendShapeLoss=False,
    exp_weight=1.0,
    rot_weight=1.0,
    trans_weight=1.0,
):
    """function that computes the various components of the VQ loss"""
    if LLoss:
        exp_loss = exp_weight * F.l1_loss(pred[:, :, :52], target[:, :, :52])
        rot_loss = rot_weight * F.l1_loss(pred[:, :, 52:55], target[:, :, 52:55])
        trans_loss = trans_weight * F.l1_loss(pred[:, :, 55:], target[:, :, 55:])
        ## loss is VQ reconstruction + weighted pre-computed quantization loss
        return (exp_loss + rot_loss + trans_loss), quant_loss.mean() * quant_loss_weight
        # the LLloss focus more about rotation, jaw and expression which make the output random generate mounth and eye movement

    elif BlendShapeLoss:
        # https://github.com/LizhenWangT/FaceVerse/issues/7
        brow_loss = F.l1_loss(pred[:, :, :5], target[:, :, :5])
        cheek_loss = F.l1_loss(pred[:, :, 5:8], target[:, :, 5:8])
        eye_blink_loss = F.l1_loss(pred[:, :, 8:10], target[:, :, 8:10])
        eye_look_loss = F.l1_loss(pred[:, :, 10:18], target[:, :, 10:18])
        eye_squint_loss = F.l1_loss(pred[:, :, 18:20], target[:, :, 18:20])
        eye_wide_loss = F.l1_loss(pred[:, :, 20:22], target[:, :, 20:22])
        jaw_loss = F.l1_loss(pred[:, :, 22:26], target[:, :, 22:26])
        mounth_first_loss = F.l1_loss(pred[:, :, 26:43], target[:, :, 26:43])
        smile_loss = F.l1_loss(pred[:, :, 43:45], target[:, :, 43:45])
        mounth_second_loss = F.l1_loss(pred[:, :, 45:49], target[:, :, 45:49])
        nose_tongue_loss = F.l1_loss(pred[:, :, 49:52], target[:, :, 49:52])
        rot_loss = F.l1_loss(pred[:, :, 52:55], target[:, :, 52:55])
        trans_loss = F.l1_loss(pred[:, :, 55:], target[:, :, 55:])
        return (
            brow_loss
            + cheek_loss
            + eye_blink_loss
            + eye_look_loss
            + eye_squint_loss
            + eye_wide_loss
            + jaw_loss
            + mounth_first_loss
            + smile_loss
            + mounth_second_loss
            + nose_tongue_loss
            + rot_loss
            + trans_loss
        ), quant_loss.mean() * quant_loss_weight
    else:
        return nn.L1Loss()(pred, target), quant_loss.mean() * quant_loss_weight


class VQPretrainer(BaseTrainer):
    def __init__(
        self,
        config: FaceTokenizerConfig,
        is_conv=False,
        render: Render = None,
        *args,
        **kwargs,
    ):
        super().__init__(config, calc_vq_loss, *args, **kwargs)
        if not is_conv:
            self.model = FaceTokenizerTransformer(config)
        else:
            self.model = FaceTokenizerANN(config, calc_vq_loss)

        self.is_conv = is_conv

        self.config = config
        self.learning_rate = config.learning_rate
        self.block_size = config.block_size
        self.render = render

        self.out_dir = config.output_dir
        self.run_name = config.run_name

        if self.config.phase == 1:
            self.phase_1()
        elif self.config.phase == 2:
            self.phase_2()

    def phase_1(self):
        # self.model.unfreeze_tokenizer()
        # self.model.freeze_decoder_emotion()
        pass

    def phase_2(self):
        self.model.freeze_tokenizer()
        self.model.unfreeze_decoder_emotion()

    def _common_step(self, batch, batch_idx):
        (
            x_hat_emotion,
            src_audio_clip,
            src_emotion,
            src_3dmm,
            x_hat_emotion,
            trg_audio_clip,
            trg_emotion,
            trg_3dmm,
            x_hat_emotion,
        ) = batch

        trg_3dmm = trg_3dmm.float()
        trg_emotion = trg_emotion.float()

        # random sample from trg_3dmm to a sequence of 32 to match the input of the model
        # rand_i = np.random.randint(0, trg_3dmm.shape[1] // self.block_size - 1)

        # predict 1 block at a time
        # trg_3dmm = trg_3dmm[
        #     :, rand_i * self.block_size : (rand_i + 1) * self.block_size, :
        # ]

        # flaten long trg_3dmm to many batch of block_size
        mul = trg_3dmm.shape[1] // self.block_size
        trg_3dmm = trg_3dmm.reshape(
            trg_3dmm.shape[0] * mul, self.block_size, trg_3dmm.shape[-1]
        )

        trg_emotion = trg_emotion.reshape(
            trg_emotion.shape[0] * mul, self.block_size, trg_emotion.shape[-1]
        )

        # original paper only take in target 3dmm
        x_hat_3dmm, x_hat_emotion, intermediate_loss = self.model(trg_3dmm)
        # print(x_hat.shape, trg_3dmm.shape)
        # quit()
        if intermediate_loss is None:
            intermediate_loss = torch.zeros(1)
            intermediate_loss = intermediate_loss.to(x_hat_3dmm.device)
            # print("intermediate loss is None")

        _3dmmloss, quant_loss = self.loss_fn(
            pred=x_hat_3dmm,
            target=trg_3dmm,
            quant_loss=intermediate_loss,
            quant_loss_weight=self.config.quantize_weight,
            LLoss=False,
            BlendShapeLoss=True,
            exp_weight=self.config.exp_weight,
            rot_weight=self.config.rot_weight,
            trans_weight=self.config.trans_weight,
        )

        _emotion_loss = F.l1_loss(x_hat_emotion, trg_emotion)

        return _3dmmloss, _emotion_loss, intermediate_loss, x_hat_3dmm, x_hat_emotion

    def training_step(self, batch, batch_idx):
        _3dmmloss, _emotion_loss, intermediate_loss, _, _ = self._common_step(
            batch, batch_idx
        )
        self.log(
            "train_face_loss", _3dmmloss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "train_emo_loss", _emotion_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "train_vq_loss",
            intermediate_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        if (
            self.trainer.current_epoch / self.trainer.max_epochs > 0.8
            and self.config.phase == 1
        ):
            self.config.phase = 2
            self.phase_2()

        if self.config.phase == 1:
            loss = (
                _3dmmloss
                + intermediate_loss.to(_3dmmloss.device) * self.config.quantize_weight
                + _emotion_loss
            )
        elif self.config.phase == 2:
            loss = _emotion_loss
        else:
            loss = (
                _3dmmloss
                + intermediate_loss.to(_3dmmloss.device) * self.config.quantize_weight
                + _emotion_loss
            )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if self.global_step < self.config.warmup_iters:
            _, scheduler = self.lr_schedulers()
            scheduler.step()

        return loss

    def validation_step(self, batch, batch_idx):
        _3dmmloss, _emotion_loss, intermediate_loss, _, _ = self._common_step(
            batch, batch_idx
        )
        self.log("val_face_loss", _3dmmloss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "val_vq_loss", intermediate_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "val_emo_loss", _emotion_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        loss = (
            _3dmmloss
            + intermediate_loss.to(_3dmmloss.device) * self.config.quantize_weight
            + _emotion_loss
        )
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # if batch_idx % 25 == 0:
        (
            _3dmmloss,
            _emotion_loss,
            intermediate_loss,
            x_hat_3dmm,
            x_hat_emotion,
        ) = self._common_step(batch, batch_idx)
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

        mul = lt_3dmm.shape[1] // self.block_size

        x_hat_3dmm = x_hat_3dmm.reshape(
            lt_3dmm.shape[0], self.block_size * mul, x_hat_3dmm.shape[-1]
        )

        batch_size = x_hat_3dmm.shape[0]
        for bs in range(batch_size):
            self.render.rendering_for_fid(
                os.path.join(self.out_dir, self.run_name),
                "{}_b{}_ind{}".format("val", str(batch_idx + 1), str(bs + 1)),
                x_hat_3dmm[bs],
                lt_video_clip[bs],
                lt_ref_image[bs],
                lt_video_clip[bs],
            )

    def configure_optimizers(self):
        if self.is_conv:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.learning_rate
            )
            return optimizer
        optimizer = self.model.configure_optimizers(
            weight_decay=self.config.weight_decay,
            learning_rate=self.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            device_type=self.device,
        )

        # return optimizer

        warmup_duration = self.config.warmup_iters

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.0001,
            end_factor=1,
            total_iters=warmup_duration,
        )

        red_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=10000, min_lr=1e-6, verbose=True
        )

        lr_scheduler = {
            "scheduler": red_plateau,
            "interval": "epoch",
            "frequency": 2,
            "monitor": "val_loss",
        }

        return ([optimizer], [lr_scheduler, {"scheduler": warmup}])


def main(
    quantize_type="vq",
    test: bool = False,
    resume: bool = False,
    resume_ckpt: str = None,
    test_checkpoint_path: str = None,
    output_dir: str = None,
    dataset_path: str = None,
    run_name: str = None,
):
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    torch.set_float32_matmul_precision("medium")

    # run_name = quantize_type + "_vocab2048_252_12_12_new_loss_two_phase"
    # test_checkpoint_path = "/home/tien/playground_facereconstruction/output/quantize_vq_pretrain_3/epoch=197-step=39600.ckpt"
    # resume_ckpt = "/home/tien/playground_facereconstruction/output/quantize_vq_pretrain_3/epoch=179-step=36000.ckpt"
    # output_dir = (
    #     "/home/tien/playground_facereconstruction/output/quantize_vq_pretrain_3"
    # )
    # dataset_path = "/home/tien/playground_facereconstruction/data/react_2024"

    render = Render("cuda")

    data_args = Dataconfig(
        dataset_path=dataset_path,
        batch_size=1 if test else 8,
        num_workers=12,
        img_size=256,
        crop_size=224,
        clip_length=736 if not test else 480,  # divisible by 32
        test_extend_factor=1,
        is_render=False,
    )

    model_config = FaceTokenizerConfig(
        input_dim=58,
        output_dim=58,
        emotion_output_dim=25,
        block_size=32,
        n_embd=252,
        n_head=12,
        n_layer=12,
        quantize_type=quantize_type,
        learning_rate=1e-3,
        output_dir=output_dir,
        run_name=run_name,
        dropout=0.1,
        attn_dropout=0.1,
        resid_dropout=0.1,
        embd_pdrop=0.1,
        warmup_iters=300,
        quantize_weight=1.0,
        exp_weight=1.0,
        trans_weight=1.0,
        rot_weight=1.0,
        quantize_codebook_size=2048,  # 512 or 1024, 2048
        phase=1,  # phase 1 or 2, phase 0 mean no phase
        # config for fsq
        # according to https://arxiv.org/pdf/2309.15505.pdf
        # codebooksize - levels: 2**8~[8,6,5], 2**10~[8,5,5,5], 2**12~[7,5,5,5,5]
        quantize_levels=[8, 5, 5, 5],
        # config for lfq
        quantize_entropy_loss_weight=0.1,
        quantize_diversity_gamma=0.75,
    )

    if quantize_type == "lfq":
        model_config.n_embd = 144
        model_config.n_head = 12
        model_config.quantize_codebook_size = 2048

    model_config.data_config = data_args

    if not test:
        if resume:
            print("resume from checkpoint: ", resume_ckpt)
            model = VQPretrainer.load_from_checkpoint(
                resume_ckpt, config=model_config, is_conv=False, render=render
            )
        else:
            model = VQPretrainer(model_config, is_conv=False, render=render)

        # model = torch.compile(model)
        datamodule = ReactDataModule(
            conf=data_args,
            # only load 3dmm because only quantize 3dmm
            load_3dmm=True,
            load_audio=False,
            load_emotion=True,
            load_ref=False,
            load_video=False,
            load_raw_audio=False,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=output_dir,
            save_top_k=1,
            mode="min",
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")
        wandb_logger = WandbLogger(
            project="quantize_vq_pretrain",
            config=model_config,
            name=run_name,
        )
    else:
        model = VQPretrainer.load_from_checkpoint(
            test_checkpoint_path, config=model_config, is_conv=False, render=render
        )

    import math

    actual_batch = (
        math.ceil(data_args.clip_length / model_config.block_size)
        * data_args.batch_size
    )
    print(f"actual batch size in training: {actual_batch}")

    trainer = Trainer(
        devices=1,
        min_epochs=1,
        max_epochs=200,
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
        check_val_every_n_epoch=2 if not test else None,
        enable_checkpointing=False if test else True,
        logger=wandb_logger if not test else None,
    )

    if not test:
        # print all config
        import time

        print("start time: \t", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print("model_config: \t", model_config)
        print("data_args: \t", data_args)
        print("output_dir: \t", output_dir)
        print("run_name: \t", run_name)
        print("resume_ckpt: \t", resume_ckpt)
        print("is_resume: \t", resume)
        print("quantize_type: \t", quantize_type)
        print("is_test: \t", test)
        print("test_checkpoint_path: \t", test_checkpoint_path)
        print("output_dir: \t", output_dir)

        trainer.fit(model, datamodule=datamodule)

        # print best checkpoint
        best_model_path = checkpoint_callback.best_model_path
        print("best model path: ", best_model_path)
        # write to file
        import datetime

        with open(output_dir + "/best_model_path.txt", "a") as f:
            f.write(quantize_type + "\n")
            f.write(best_model_path)
            f.write("\nfinished_time:" + str(datetime.datetime.now()) + "\n")

        # log path to wandb
        wandb_logger.log_hyperparams({"best_model_path": best_model_path})
        test_checkpoint_path = best_model_path

    data_args.batch_size = 1
    data_args.clip_length = 128
    data_args.is_render = True
    datamodule = ReactDataModule(
        conf=data_args,
        # only load 3dmm because only quantize 3dmm
        load_3dmm=True,
        load_audio=False,
        load_emotion=True,
        load_ref=True,
        load_video=True,
        load_raw_audio=False,
    )

    trainer.test(model, ckpt_path=test_checkpoint_path, datamodule=datamodule)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", default=None, type=str, help="run name")

    parser.add_argument("--quantize_type", default="vq", type=str, help="vq, fsq, lfq")

    parser.add_argument("--resume", default=False, type=bool, help="resume training")
    parser.add_argument(
        "--resume_ckpt", default=None, type=str, help="resume checkpoint"
    )

    parser.add_argument("--test", default=False, type=bool, help="test mode")
    parser.add_argument(
        "--test_checkpoint_path", default=None, type=str, help="test ckpt"
    )

    parser.add_argument("--output_dir", default=None, type=str, help="output dir")
    parser.add_argument("--dataset_path", default=None, type=str, help="dataset path")

    args = parser.parse_args()

    main(
        run_name=args.run_name,
        quantize_type=args.quantize_type,
        test=args.test,
        resume=args.resume,
        resume_ckpt=args.resume_ckpt,
        test_checkpoint_path=args.test_checkpoint_path,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
    )

    # sample command no resume
    # python p1_face_tokenizer.py --run_name "quantize_vq_pretrain_3" --quantize_type "vq" --output_dir "/home/tien/playground_facereconstruction/output/quantize_vq_pretrain_3" --dataset_path "/home/tien/playground_facereconstruction/data/react_2024"
