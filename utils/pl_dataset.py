import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets import TrainValDataset


class MyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, gt, fraction, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.gt = gt
        self.fraction = fraction
        self.train_set = None
        self.val_set = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_set = TrainValDataset(mode='train', data_dir=self.data_dir,
                                         gt=self.gt, fraction=self.fraction)

        self.val_set = TrainValDataset(mode='val', data_dir=self.data_dir,
                                       gt=self.gt, fraction=self.fraction)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def teardown(self, stage):
        pass
