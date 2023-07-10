import os 
from argparse import ArgumentParser
import pytorch_lightning as pl

from datamodules.brain import get_brain_dataset, init_arg_dicts, get_brain_dataset_eval
from config.datasets.brain import get_brain_args
import subprocess
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
    #def __init__(self, batch_size:int, patch_size:int = (50,50,50),base_dir:list = [],*args, **kwargs):
    def __init__(self, batch_size: int = 64, double_headed: bool = False, patch_size:int = (50,50,50), input: str= 'insp', overlap: str='20', kfold: int=1, max_patches: int=None,
                 base_train='default', base_dir: list = [], mode='train', step='pretext', realworld_dataset=False, *args, **kwargs):
        super().__init__()
        #self.args = {'batch_size':batch_size, "patch_size":patch_size}
        self.args = {'batch_size':batch_size, 'double_headed':double_headed,
                    'base_train':base_train, "patch_size":patch_size, 'step': step, 'realworld_dataset': realworld_dataset, 'input': input, 'overlap': overlap, 'kfold': kfold, 'max_patches': max_patches}

        if double_headed: #or transform_type == 'split':
            val_args = {"mode": "train"}
        else:
            val_args = {}
        self.common_args = dict(**datasets_common_args)
        self.common_args.update(**self.args)
        self.train_args = dict(**datasets_train_args)
        self.train_args['base_dir'] += base_dir

        #self.train_args['mode'] = mode

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

    def val_dataloader(self):
        return get_brain_dataset(**self.common_args, **self.val_args)

    def test_dataloader(self):
        return get_brain_dataset_eval(**self.common_args, **self.test_args)

    def get_ano_loader(self):
        raise NotImplementedError

    def get_brain_datasets(self):
        return get_brain_datasets(self.common_args, self.train_args, self.val_args, 
                        self.val_ano_args, self.test_args)

    @staticmethod
    def get_shape(**kwargs):
        if kwargs['input'] in ['insp_exp_reg', 'insp_jacobian']:
            num_modalities = 2
        elif  kwargs['input'] == 'insp':
            num_modalities = 1
        else:
            NotImplementedError
        return (num_modalities, *kwargs['target_size'])

    def get_workers_for_current_node(self) -> int:
        num_workers: int
        hostname = subprocess.getoutput(["hostname"])  # type: ignore

        if hostname in ["hdf19-gpu16", "hdf19-gpu17", "hdf19-gpu18", "hdf19-gpu19"]:
            num_workers = 16
        if hostname.startswith("hdf19-gpu") or hostname.startswith("e071-gpu"):
            num_workers = 12
        elif hostname.startswith("e230-dgx1"):
            num_workers = 10
        elif hostname.startswith("hdf18-gpu"):
            num_workers = 16
        elif hostname.startswith("e230-dgx2"):
            num_workers = 6
        elif hostname.startswith(("e230-dgxa100-", "lsf22-gpu", "e132-comp")):
            num_workers = 32
        elif hostname.startswith("e230-pc24"):  # Own workstation
            num_workers = 24
        else:
            raise NotImplementedError()
        return num_workers

    @staticmethod
    def add_data_specific_args(parent_parser):
        #Dataset specific arguments
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--input", default='insp', type=str, choices=['insp', 'insp_exp_reg', 'insp_jacobian', 'jacobian'])
        parser.add_argument("--overlap", default='20', type=str, choices=['0', '20'])
        parser.add_argument("--kfold", default=1, type=int)
        parser.add_argument("--max_patches", default=None, type=int)
        parser.add_argument("--train_exposure", default=None, type=str)
        parser.add_argument("--batch_size", default=8, type=int)#64
        parser.add_argument("--mask_type", default=None, type=str)
        parser.add_argument("--resume", type=str, default= None) # '/home/silvia/Documents/CRADL/logs_cradl/copdgene/pretext/brain/simclr-resnet18/default/17030592/checkpoints/epoch=16-step=133585.ckpt')

        #Training specific arguments
        #parser.add_argument("--num_workers", default=8, type=int) #8 #12
        parser.add_argument("--dataset", default='brain', type=str) # choices=['brain'],type=str)
        parser.add_argument("--target_size", default=(50,50,50), type=int)
        parser.add_argument("--base_train", default='models_genesis', type=str, choices=['default', 'models_genesis'])
        parser.add_argument("--step", default='pretext', type=str, choices=['pretext', 'fitting_GMM', 'eval', 'test'])
        parser.add_argument("--realworld_dataset", default=False, type=bool)
        parser.add_argument("--default_experiment", default='simclr', type=str, choices=['simclr', 'nnclr'])

        return parser
        
