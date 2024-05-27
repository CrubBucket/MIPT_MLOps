import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import hydra
from omegaconf import DictConfig

from pl_dataset import MyDataModule
from pl_training_module import MyTrainingModule
from model import Model


@hydra.main(config_path="../config", config_name="conf", version_base="1.3")
def main(cfg: DictConfig):
    print('GPU available - ' + str(torch.cuda.is_available()), 10 * '-')
    pl.seed_everything(cfg.params.SEED)

    data_module = MyDataModule(
        data_dir=cfg.data.images_path,
        gt=cfg.data.gt_path,
        fraction=cfg.data.fraction,
        batch_size=cfg.data.batch_size,
    )

    model = Model(image_channels=cfg.model.image_channels,
                  num_classes=cfg.model.num_classes)

    training_module_ckpt = ModelCheckpoint(
        dirpath=cfg.callbacks.checkpoint.dirpath,
        filename=cfg.callbacks.checkpoint.filename,
        monitor=cfg.callbacks.checkpoint.monitor,
        mode=cfg.callbacks.checkpoint.mode,
        save_top_k=cfg.callbacks.checkpoint.save_top_k,
    )

    early_stopping = EarlyStopping(monitor=cfg.callbacks.early_stop.monitor,
                                   mode=cfg.callbacks.early_stop.mode,
                                   patience=cfg.callbacks.early_stop.patience)

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=[early_stopping, training_module_ckpt],
        log_every_n_steps=cfg.training.log_every_n_steps,
    )

    training_module = MyTrainingModule(model=model)

    trainer.fit(model=training_module, datamodule=data_module)


if __name__ == "__main__":
    main()

