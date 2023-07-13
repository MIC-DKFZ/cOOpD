import logging
import os

import pytorch_lightning as pl

from algo.model import get_args, get_experiment
from config.paths import glob_conf
from datamodules.lung_module import LungDataModule
from pytorch_lightning.loggers import tensorboard
## To exchange Datasets - Exchange LungDataModule for another PL Datamodule


def main(args):
    exp_name = args.experiment +"-"+ args.model_type
    

    if args.experiment in ['simclr', 'nnclr']:
        args.double_headed=True
        if args.mask_type is None:
            args.mask_type='noise'

    
    else:
        raise NotImplementedError

    if args.experiment == 'flow':
        args.gradient_clip_val = 10.

    if args.resume:
        args.resume_from_checkpoint = args.resume


    # Initialize logger
    logdir = os.path.join(glob_conf['logpath'], 'pretext', args.dataset, exp_name)


    logger = pl.loggers.TensorBoardLogger(logdir, name=args.name, version=args.version)

    # Initialize Trainer & Write logger to Trainer
    ckpt_callback = pl.callbacks.ModelCheckpoint(monitor='val/loss', mode='min') #, every_n_train_steps=10, save_last=True) #delete every_n_train_steps and save_last
    trainer = pl.Trainer.from_argparse_args(args, logger=logger,callbacks=[ckpt_callback])


    # Initialize Data
    dm = LungDataModule(**vars(args))
    dm.prepare_data()
    args.num_workers = dm.get_workers_for_current_node()
    print(args)

    train_loader, val_loader = dm.train_dataloader()

    #loader_dict = {'train':dm.train_dataloader()[0], 'val':dm.train_dataloader()[1], 'test':dm.test_dataloader()}
    loader_dict = {'train':train_loader, 'val':val_loader, 'test':dm.test_dataloader()}

    args.num_samples = len(loader_dict['train'])*args.batch_size

    #Initialize Model
    experiment = get_experiment(args)
    trainer.fit(model=experiment, train_dataloader=loader_dict['train'], val_dataloaders=loader_dict['val'])
    print(logger.log_dir)
    #return experiment.logger.log_dir
    return logger.log_dir



if __name__ == "__main__":
    args = get_args(LungDataModule)
    path = main(args)
    print(path)