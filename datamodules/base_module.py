from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from datamodules.utils import MapDataset


class BaseDatamodule(pl.LightningDataModule):
    def __init__(self, acq_data ,data_root, batch_size, pin_memory, transform_train, transform_val, num_workers, drop_last ,
                transform_target, val_size, random_split=False,*args, **kwargs):
        super(BaseDatamodule, self).__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.transform_train = transform_train
        self.transform_val = transform_val
        self.transform_target = transform_target
        self.val_size = val_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.random_split = random_split

        self.acq_data = acq_data


    def prepare_data(self):
        # download
        self.acq_data(self.data_root, train=True, download=True)
        self.acq_data(self.data_root, train=False, download=True)

    def setup(self, stage=None):
        if stage in ['fit' ,'train', 'val', None]:
            data_full = self.acq_data(self.data_root, train=True, transform=None)
            length_full = len(data_full)
            length_val = int((length_full*self.val_size))
            length_train = int(length_full-length_val)
            if self.random_split:
                self.train, self.val = torch.utils.data.random_split(data_full, [length_train, length_val])
            else:
                self.train, self.val = data_full[:length_train], data_full[length_train:]
            self.train = MapDataset(self.train, transform=self.transform_train, transform_target=self.transform_target)
            self.val = MapDataset(self.val, transform=self.transform_val, transform_target=self.transform_target)
            self.num_samples = len(self.train) 
        if stage in ['test', None]:
            self.test = self.acq_data(self.data_root, train=False, transform=self.transform_val, target_transform=self.transform_target)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, drop_last=self.drop_last, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, drop_last=self.drop_last, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, drop_last=self.drop_last, num_workers=self.num_workers, pin_memory=self.pin_memory)

    @staticmethod
    def add_data_specific_args(parent_parser):
        #Dataset specific arguments
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", default=15, type=int)
        parser.add_argument("--input", default='insp', type=str, choices=['insp', 'insp_exp_reg', 'insp_jacobian', 'jacobian'])
        #Training specific arguments
        parser.add_argument("--val_size", default=0.1, type=float)
        parser.add_argument("--num_workers", default=8, type=int)
        return parser

