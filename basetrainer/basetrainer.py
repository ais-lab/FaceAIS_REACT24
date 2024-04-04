import lightning as pl


class BaseTrainer(pl.LightningModule):
    def __init__(self, model, loss_fn=None, *args, **kwargs):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch, batch_idx)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch, batch_idx)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        return loss

    def _common_step(self, batch, batch_idx):
        (
            _,
            src_audio_clip,
            src_emotion,
            src_3dmm,
            _,
            trg_audio_clip,
            trg_emotion,
            trg_3dmm,
            _,
        ) = batch

        src_3dmm = src_3dmm.float()
        trg_3dmm = trg_3dmm.float()
        src_audio_clip = src_audio_clip.float()
        trg_audio_clip = trg_audio_clip.float()

        # input of model should be (src_audio_clip, src_3dmm)
        x_hat, intermediate_loss = self((src_audio_clip, src_3dmm))

        loss = self.loss_fn(
            pred=x_hat, target=trg_3dmm, intermediate_loss=intermediate_loss
        )

        return loss, x_hat, batch

    def configure_optimizers(self):
        raise NotImplementedError
