import os 
from argparse import ArgumentParser
import pytorch_lightning as pl

from datamodules.brain import get_brain_dataset, init_arg_dicts
from config.datasets.brain import get_brain_args

## Readout config values
brain_base_args = get_brain_args(mode='train')
## Args for Datasets used for the contrastive Training (& the generative models)!
datasets_common_args = brain_base_args['common_args']
datasets_train_args = brain_base_args['trainset_args']
datasets_val_args = brain_base_args['valset_args']

## Args for Validation Datasets containing anomalies which are labelled!
datasets_val_ano_args = brain_base_args['valanoset_args']
datasets_test_args = brain_base_args['testset_args']



def get_brain_datasets(common_args=None, trainset_args=None, valset_args=None, valanoset_args=None, testset_args=None):
    common_args, trainset_args, valset_args, valanoset_args, testset_args = init_arg_dicts(common_args, trainset_args,
                                                                                           valset_args, valanoset_args,
                                                                                           testset_args)
    d_common_args = dict(**datasets_common_args)
    d_common_args.update(common_args)

    d_train_args = dict(**datasets_train_args)
    d_train_args.update(trainset_args)

    d_val_args = dict(**datasets_val_args)
    d_val_args.update(valset_args)

    d_val_ano_args = dict(**datasets_val_ano_args)
    d_val_ano_args.update(valanoset_args)

    d_test_args = dict(**datasets_test_args)
    d_test_args.update(testset_args)

    train_loader, val_loader = get_brain_dataset(**d_common_args, **d_train_args)
    #val_loader = get_brain_dataset(**d_common_args, **d_val_args)
    val_ano_loader = get_brain_dataset(**d_common_args, **d_val_ano_args)
    test_loader = get_brain_dataset(**d_common_args, **d_test_args)

    return {"train": train_loader, "val": val_loader, "val-ano": val_ano_loader, "test": test_loader, "n_channels": 1}


class BrainDataModule(pl.LightningDataModule):
    def __init__(self, batch_size:int, double_headed:bool=False, mask_type:str = 'test', target_size:int = 128, transform_type='single', base_train='default',base_dir:list = [], mode='train',*args, **kwargs):
        super().__init__()
        self.args = {'batch_size':batch_size, "patch_size":patch_size}

        val_args= {}
        self.common_args = dict(**datasets_common_args)
        self.common_args.update(**self.args)
        self.train_args = dict(**datasets_train_args)
        self.train_args['base_dir'] += base_dir

        self.train_args['mode'] = mode

        self.val_args = dict(**datasets_val_args)
        self.val_args.update(val_args)
        self.val_ano_args = dict(**datasets_val_ano_args)
        self.test_args = dict(**datasets_test_args)
        


    def prepare_data(self):
        pass 

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return get_brain_dataset(**self.common_args, **self.train_args)

    # def val_dataloader(self):
    #     return get_brain_dataset(**self.common_args, **self.val_args)

    # def test_dataloader(self):
    #     return get_brain_dataset(**self.common_args, **self.test_args)

    def get_ano_loader(self):
        raise NotImplementedError

    def get_brain_datasets(self):
        return get_brain_datasets(self.common_args, self.train_args, self.val_args, 
                        self.val_ano_args, self.test_args)

    @staticmethod
    def get_shape(**kwargs):
        num_modalities = 1 #change
        return (num_modalities, *kwargs['target_size'])

    @staticmethod
    def add_data_specific_args(parent_parser):
        #Dataset specific arguments
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--input", default='insp', type=str, choices=['insp', 'insp_exp_reg', 'insp_jacobian', 'jacobian'])
        parser.add_argument("--train_exposure", default=None, type=str)
        parser.add_argument("--batch_size", default=64, type=int)
        parser.add_argument("--mask_type", default=None, type=str)
        #Training specific arguments
        parser.add_argument("--num_workers", default=12, type=int)
        parser.add_argument("--dataset", default='brain', type=str) # choices=['brain'],type=str)
        parser.add_argument("--target_size", default=128, type=int)
        parser.add_argument("--base_train", default='default', type=str)
        return parser
        
