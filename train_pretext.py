import os

import pytorch_lightning as pl

from algo.model import get_args, get_experiment
from config.paths import glob_conf
from datamodules.brain_module import BrainDataModule

## To exchange Datasets - Exchange BrainDataModule for another PL Datamodule


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


    # Initialize logger
    logdir = os.path.join(glob_conf['logpath'], 'pretext', args.dataset, exp_name)


    logger = pl.loggers.TensorBoardLogger(logdir, name=args.name, version=args.version)

    # Initialize Trainer & Write logger to Trainer
    ckpt_callback = pl.callbacks.ModelCheckpoint(monitor='val/loss', mode='min')
    trainer = pl.Trainer.from_argparse_args(args, logger=logger,callbacks=[ckpt_callback])


    # Initialize Data
    dm = BrainDataModule(**vars(args))
    dm.prepare_data()
    

    loader_dict = {'train':dm.train_dataloader(), 'val':dm.val_dataloader(), 'test':dm.test_dataloader()}
    args.num_samples = len(loader_dict['train'])*args.batch_size

    #Initialize Model
    experiment = get_experiment(args)


    trainer.fit(model=experiment, train_dataloader=loader_dict['train'], val_dataloaders=loader_dict['val'])
    return experiment.logger.log_dir


if __name__ == "__main__":
    args = get_args(BrainDataModule)
    path = main(args)
    print(path)