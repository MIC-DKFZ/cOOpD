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
from algo.utils import get_seg_im_gt, process_batch
from config.paths import trainer_defaults
from pytorch_lightning.utilities import argparse_utils


############## Possibly Rewrite this function obtaining y $#####################
def get_label_latent(experiment, dataloader, get_slice=False, to_npy=False, num_epoch=1):
    experiment.freeze()
    latent = []
    labels = []
    slices = []
    with torch.no_grad():
        for epoch in range(num_epoch):
            for i, batch in enumerate(dataloader):
                x, y = process_batch(batch)
                # try:
                #     from batchviewer import view_batch
                #     view_batch(batch['data'][0][0])
                #     view_batch(batch['data'][5][0])
                #     view_batch(batch['data'][10][0])
                #     view_batch(batch['data'][15][0])
                #     view_batch(batch['data'][20][0])
                #     view_batch(batch['data'][25][0])
                #     view_batch(batch['data'][35][0])
                #     view_batch(batch['data'][45][0])
                #     view_batch(batch['data'][55][0])
                #     view_batch(batch['data'][11][0])
                #     view_batch(batch['data'][37][0])
                #     view_batch(batch['data'][58][0])
                #     view_batch(batch['data'][63][0])
                # except ImportError:
                #     view_batch = None
                if y is not None and len(y.shape)>1:
                    #y, _ = get_seg_im_gt(batch)
                    y = batch['label']
                x = x.to(experiment.device)
                z = experiment.encode(x)
                latent.append(z.detach()) 
                latent= [torch.cat(latent, dim=0)]
                labels.append(y.detach()) 
                labels= [torch.cat(labels, dim=0)]
                if get_slice:
                    slices.append(batch['slice_idxs'].detach())
                    slices = [torch.cat(slices, dim=0)]

    
    labels = labels[0].to('cpu')
    latent = latent[0].to('cpu')
    if to_npy:
        labels=labels.numpy()
        latent=latent.numpy()
    
    out_dict = {'labels' : labels, 'latent' : latent}
    if get_slice:
        slices = slices[0].to('cpu')
        if to_npy:
            slices = slices.numpy()
        out_dict['slice_idxs'] = slices
    return out_dict







def get_label_latent_forCNN(experiment, dataloader, get_slice=False, to_npy=False, num_epoch=1):
    experiment.freeze()
    latent = []
    labels = []
    location = []
    coordinates = []
    patch_number = []
    patient = []
    slices = []
    with torch.no_grad():
        for epoch in range(num_epoch):
            for i, batch in enumerate(dataloader):
                x, y = process_batch(batch)
                # try:
                #     from batchviewer import view_batch
                #     view_batch(batch['data'][0][0])
                #     view_batch(batch['data'][5][0])
                #     view_batch(batch['data'][10][0])
                #     view_batch(batch['data'][15][0])
                #     view_batch(batch['data'][20][0])
                #     view_batch(batch['data'][25][0])
                #     view_batch(batch['data'][35][0])
                #     view_batch(batch['data'][45][0])
                #     view_batch(batch['data'][55][0])
                #     view_batch(batch['data'][11][0])
                #     view_batch(batch['data'][37][0])
                #     view_batch(batch['data'][58][0])
                #     view_batch(batch['data'][63][0])
                # except ImportError:
                #     view_batch = None
                if y is not None and len(y.shape) > 1:
                    # y, _ = get_seg_im_gt(batch)
                    y = batch['label']
                    loc = torch.as_tensor(list(map(int, batch['metadata']['location'])))
                    coord = torch.as_tensor(batch['metadata']['coordinates'])
                    patch_num = torch.as_tensor(batch['metadata']['patch_num'])
                    pat_ID = torch.as_tensor(list(map(int, [x[:-1] for x in batch['patient_name']])))
                x = x.to(experiment.device)
                z = experiment.encode(x)
                latent.append(z.detach())
                latent = [torch.cat(latent, dim=0)]
                labels.append(y.detach())
                labels = [torch.cat(labels, dim=0)]
                location.append(loc)
                location = [torch.cat(location, dim = 0)]
                coordinates.append(coord)
                coordinates = [torch.cat(coordinates, dim = 0)]
                patch_number.append(patch_num.detach())
                patch_number = [torch.cat(patch_number, dim = 0)]
                patient.append(pat_ID)
                patient = [torch.cat(patient, dim = 0)]
                if get_slice:
                    slices.append(batch['slice_idxs'].detach())
                    slices = [torch.cat(slices, dim=0)]

    labels = labels[0].to('cpu')
    latent = latent[0].to('cpu')
    location = location[0].to('cpu')
    coordinates = coordinates[0].to('cpu')
    patch_number = patch_number[0].to('cpu')
    patient = patient[0].to('cpu')
    if to_npy:
        labels = labels.numpy()
        latent = latent.numpy()
        location = location.numpy()
        coordinates = coordinates.numpy()
        patch_number = patch_number.numpy()
        patient = patient.numpy()

    out_dict = {'labels': labels, 'latent': latent, 'location': location, 'coordinates': coordinates, 'patch_number': patch_number, 'patient': patient}
    if get_slice:
        slices = slices[0].to('cpu')
        if to_npy:
            slices = slices.numpy()
        out_dict['slice_idxs'] = slices

    return out_dict


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

def get_experiment(args, num_layers:int = 5): #2


    if args.experiment in ['simclr']:
        if args.model_type[:6] == 'CNN3D':
            model = base.CNN3D(in_channels=args.input_shape[0], out_channels=args.z_dim, num_feat=args.num_ft, bias=True, num_layers=num_layers)
        elif args.model_type[:6] == 'resnet':
            args.num_ft = 0
            if args.input_shape[1] == 50:
                cifar_stem=True
            else:
                raise NotImplementedError
            model = base.ResNet_Encoder(base_model=args.model_type, channels_in=args.input_shape[0], cifar_stem=cifar_stem)
        elif args.model_type[:6] == 'VGG11':
            model = base.VGG(in_channels= args.input_shape[0], vgg_name='VGG11')
        elif args.model_type[:6] == 'VGG13':
            model = base.VGG(in_channels= args.input_shape[0], vgg_name='VGG13')
        elif args.model_type[:6] == 'VGG16':
            model = base.VGG(in_channels= args.input_shape[0], vgg_name='VGG16')
        elif args.model_type[:6] == 'VGG19':
            model = base.VGG(in_channels= args.input_shape[0], vgg_name='VGG19')
        # Here can new model types be implemented!
        else:
            raise NotImplementedError
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
    #hparams['model_type'] = path.split('/')[-3].split('-')[1]
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

