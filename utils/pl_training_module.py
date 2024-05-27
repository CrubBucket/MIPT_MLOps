import torch
from torch import nn
import pytorch_lightning as pl
from model import Model


class MyTrainingModule(pl.LightningModule):
    def __init__(self, model=None):
        super().__init__()
        if model:
            self.model = model
        else:
            self.model = Model()
        self.loss = nn.MSELoss()

    def training_step(self, batch, batch_idx):

        x, y = batch

        outp = self.model(x)
        loss = self.loss(outp, y)

        metrics = {'train_loss': loss}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        lr_dict = {"scheduler": lr_scheduler,}

        return [optimizer], [lr_dict]

    def validation_step(self, batch, batch_idx):

        x, y = batch

        outp = self.model(x)
        loss = self.loss(outp, y)

        metrics = {"val_loss": loss}
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return metrics
