import torch 
import os
import yaml
import numpy as np
import pytorch_lightning as pl
from argparse import Namespace, ArgumentParser
import math
import random

from models import base 


from algo.base_algo.simclr import SimCLR_base

from config.paths import trainer_defaults


def get_args(DataModule:pl.LightningDataModule, default_experiment='simclr' ,arguments=None):
    """Helper to obtain the arguments for the features training

    Args:
        DataModule (pl.LightningDataModule): [description]

    Raises:
        Exception: [description]

    Returns:
        [Namespace]: Arguments
    """
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--experiment', default=default_experiment, type=str)
    parser = DataModule.add_data_specific_args(parser)

    args_temp = parser.parse_known_args()[0]
    

    if args_temp.experiment == 'simclr':
        Experiment = SimCLR_base
    
    
    else:
        raise Exception("Invalid Experiment Choice!")
    args = get_args_exp(Experiment, DataModule, experiment_name=args_temp.experiment, arguments=arguments)

    return args

def get_base_parser(default_experiment=None):
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--name', default='default', type=str)
    parser.add_argument('--experiment', default=default_experiment,type=str)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument("--version", type=str, default=None)

    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(**trainer_defaults)
    return parser

def finalize_args(args, Experiment:pl.LightningModule ,DataModule:pl.LightningDataModule):
    args.input_shape = DataModule.get_shape(**vars(args)) 
    if args.seed is None:
        args.seed = random.randint(0, 360)
    pl.seed_everything(args.seed)
    if 'LSB_JOBID' in os.environ.keys():
        args.version = os.environ['LSB_JOBID']
    return args


def get_args_exp(Experiment:pl.LightningModule ,DataModule:pl.LightningDataModule, experiment_name='exp_name', arguments=None):
    """Parses arguments given the Experiment and DataModule, if arguments a sequence, these are parsed

    Args:
        Experiment (pl.LightningModule): [description]
        DataModule (pl.LightningDataModule): [description]
        experiment_name (str, optional): [description]. Defaults to 'exp_name'.
        arguments ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    parser = get_base_parser(experiment_name)
    parser = DataModule.add_data_specific_args(parser)
    parser = Experiment.add_model_specific_args(parser)
    
    args = parser.parse_args(arguments)

    args = finalize_args(args, Experiment, DataModule)

    return args


def get_decoder(args, num_layers:int = 0, final=torch.nn.Sigmoid):
    if num_layers == 0:
        num_layers = int(math.log2(args.input_shape[-1])//1)-1
    if (math.pow(2, num_layers+1) / args.input_shape[-1])!= 1 and args.model_type=='dc':
        message = "The input size of {} cannot be used for a DC Enc/Dec based model \nTry 2**x = size".format(args.input_shape[-1])
        raise Exception(message)
    if args.model_type == 'dc':
        from models.base import Decoder 
        return Decoder(args.z_dim, args.input_shape[0], args.num_ft, num_layers=num_layers, bias=True, final=final)
    else:
        raise NotImplementedError

def get_experiment(args, num_layers:int = 0):


    if num_layers == 0:
        num_layers = int(math.log2(args.input_shape[-1])//1)-1

    if (math.pow(2, num_layers+1) / args.input_shape[-1])!= 1 and args.model_type=='dc':
        message = "The input size of {} cannot be used for a DC Enc/Dec based model \nTry 2**x = size".format(args.input_shape[-1])
        raise Exception(message)


    if args.experiment in ['simclr']:
        if args.model_type[:6] == 'resnet':
            args.num_ft = 0
            if args.input_shape[1] == 32:
                cifar_stem=True
            elif args.input_shape[1] >= 64:
                cifar_stem = False
            else:
                raise NotImplementedError
            model = base.ResNet_Encoder(base_model=args.model_type, channels_in=args.input_shape[0], cifar_stem=cifar_stem)
        elif args.model_type == 'dc':
            model = base.get_encoder(in_channels=args.input_shape[0], out_channels=args.z_dim, num_feat=args.num_ft, bias=True, num_layers=num_layers)
        # Here can new model types be implemented!
        else:
            raise NotImplementedError
        args.z_dim = model.z_dim
    else:
        raise NotImplementedError

    # Initialize Experiment
    if args.experiment in ['simclr', 'binary']:
        experiment = SimCLR_base(hparams=args, model=model, **vars(args))

    return experiment



def get_best_checkpoint(path):
    #TODO: Make this function more generalizable
    for ckpt in os.listdir(os.path.join(path,'checkpoints')):
        if ckpt[:5] =='epoch':
            return os.path.join(path, 'checkpoints', ckpt)

    print("No Correspoinding checkpoint found using: checkpoint_final.ckpt")
    return os.path.join(path, 'checkpoints', 'checkpoint_final.ckpt')

def get_hparams(path):
    with open(os.path.join(path, 'hparams.yaml'), 'r') as file:
        hparams = yaml.load(file, Loader=yaml.FullLoader)
    hparams = Namespace(**hparams)
    return hparams




def load_best_model(loading_path):
    args = get_hparams(loading_path)
    checkpoint_path =  get_best_checkpoint(loading_path)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    # args.num_ft = args_backbone.num_ft
    # args.z_dim = args_backbone.z_dim
    # args.model_type = args_backbone.model_type

    #Initialize Model
    experiment = get_experiment(args)
    try:
        experiment(torch.randn([1, *args.input_shape]))
    except:
        pass
    experiment.load_state_dict(checkpoint['state_dict'])
    experiment.to('cpu')
    return experiment, args