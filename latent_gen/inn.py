import os

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml
from algo.model import get_hparams 
from config.paths import glob_conf
from torch.utils.data import DataLoader, TensorDataset

from latent_gen.ood_model import Abstract_OOD

from pyutils.py import is_cluster

def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(),
                         nn.Linear(512,  c_out))

def basic_FcInn(inpt, num_blocks=8):
    nodes = [Ff.InputNode(inpt, name='input')]

    # Use a loop to produce a chain of coupling blocks
    for k in range(num_blocks):
        nodes.append(Ff.Node(nodes[-1],
                             Fm.GLOWCouplingBlock,
                             {'subnet_constructor':subnet_fc, 'clamp':2.0},
                             name=F'coupling_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed':k},
                             name=F'permute_{k}'))

    nodes.append(Ff.OutputNode(nodes[-1], name='output'))
    inn = Ff.ReversibleGraphNet(nodes)
    return inn

class INN_exp(pl.LightningModule):
    def __init__(self, inpt, eps=1e-3, learning_rate=1e-3):
        super().__init__()
        self.inpt_size = [inpt]
        self.model = basic_FcInn(inpt)
        # self.in_means = None
        # self.in_stds = None
        self.eps=eps
        self.init_buffers()
        self.learning_rate = learning_rate


    def init_buffers(self):
        if len(self.inpt_size)==1:
            shape = [1, self.inpt_size[0]]
        elif len(self.inpt_size) == 3:
            shape = [1, self.inpt_size[0], 1 , 1] 
        else:
            raise NotImplementedError
        self.register_buffer('in_means', torch.zeros(size=shape, device=self.device))
        self.register_buffer('in_stds', torch.ones(size=shape, device=self.device))

    def init_preprocessing(self, in_means:torch.Tensor, in_stds:torch.Tensor, eps=1e-5):
        assert(in_means.shape == self.in_means.shape) # Shape of means is not equal
        assert(in_stds.shape == self.in_stds.shape) # Shape of stds is not equal
    
        self.in_means[:] = in_means.to(self.device) 
        self.in_stds[:] = in_stds.to(self.device) + eps
        # self.in_stds = self.in_stds + 1e-5
        
    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, [20, 40], gamma=0.1
            )

        return [optimizer], [scheduler]

    def forward(self, x, inference=True):
        x = (x-self.in_means)/self.in_stds
        if inference is False:
            x += torch.randn(x.shape, device=x.device)*self.eps
        z = self.model.forward(x)
        return z 
        
        
    def training_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        loss, _ = self.shared_step(x)
        self.log("train/loss", loss.detach())
        return loss
    
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        loss, _ =  self.shared_step(x)
        self.log("val/loss", loss.detach())

    
    def test_step(self, batch, batch_idx, *args, **kwargs):
        x, y = batch
        loss, _ =  self.shared_step(x)
        self.log( "test/loss", loss.detach())

        
    def shared_step(self, x):
        z = self.forward(x, inference=False)
        loss = 0.5*(torch.pow(z, 2)/2).sum(dim=1) - self.model.log_jacobian(run_forward=False)
        loss = loss.mean()
        return loss, None
    
    def get_scores(self, x, eps=0):
        z = self.forward(x, inference=True)
        scores = 0.5*(torch.pow(z, 2)).sum(dim=1) - self.model.log_jacobian(run_forward=False)
        return scores

    def load_state_dict(self, state_dict, strict=True):
        x = torch.randn([1, *self.inpt_size], device=self.device)
        self.model.forward(x)
        super().load_state_dict(state_dict=state_dict, strict=strict)
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        x = torch.randn([1, *self.inpt_size], device=self.device)
        self.model.forward(x)
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        return state_dict


def numpy2dataset(x:np.ndarray, y:np.ndarray, cond:np.ndarray=None, c=0, inlier=True, batch_size=64, shuffle=False):
    mask = y==c
    y[mask] = 0
    y[~mask] = 1
    if inlier is True:
        mask = y==0
        x_in = x[mask]
        y_in = y[mask]
    else:
        x_in = x
        y_in = y
        
    x_in_t = torch.Tensor(x_in)
    y_in_t = torch.Tensor(y_in)
    dataset = TensorDataset(x_in_t, y_in_t)
    dataloader =  DataLoader(dataset, num_workers=12, batch_size=batch_size, shuffle=shuffle)
    return dataloader

class INN_latent(Abstract_OOD):
    name = 'INN'
    def __init__(self, input_shape:int, path:str= None, debug=False, batch_size=512, *args, **kwargs):
        """Maximum Likelihood based INN usable in the Abstract_OOD framework. 
        Uses the Path to build a path similar to the structure of the backbone model

        Args:
            input_shape (int): [description]
            path (str, optional): [Path to backbone model]. Defaults to None.
        """
        super().__init__()
        self.model = INN_exp(inpt=input_shape).to('cuda:0')
        self.threshold= None
        self.path= path
        self.train_path= None
        self.learning_rate= 1e-3 *batch_size/256
        self.debug=debug
        self.batch_size=batch_size

    def fit(self, X:dict, Y:dict, c=0, num_epochs=60, *args, **kwargs):
        """Fits the model on the predictors X and labels Y.
        X & Y need to contain Train, (Val) & Test

        Args:
            X ([dict]): Dict of np.arrays containing predictors
            Y ([dict]): Dict of np.arrays contatining labels
            c (int, optional): [Inlier Class]. Defaults to 0.
            num_epochs (int, optional): [description]. Defaults to 60.
        """
        means = torch.Tensor(X['Train'].mean(axis=0, keepdims=True))
        stds = torch.Tensor(X['Train'].std(axis=0, keepdims=True))
        self.model.init_preprocessing(means, stds)
        Loader_in = dict()
        for key in X.keys():
            if key == "Train":
                shuffle=True 
            else:
                shuffle=False
            Loader_in[key] = numpy2dataset(X[key], Y[key], c=0, inlier=True, batch_size=self.batch_size, shuffle=shuffle)


        ckpt_callback = pl.callbacks.ModelCheckpoint(monitor='val/loss', mode='min')
        if self.path is not None:
            args_backbone = get_hparams(self.path)
            logpath = os.path.join(glob_conf['logpath'],'inn_latent', args_backbone.dataset, 
                    args_backbone.experiment +"-"+ args_backbone.model_type)

            if args_backbone.dataset == 'mvtec':
                logpath = os.path.join(logpath, args_backbone.image_class)
            try:
                name = args_backbone.name
            except:
                print("Old Model")
                name = None
        else:
            logpath = os.path.join(glob_conf['logpath'], 'debug', 'test', self.name)
            name = None
        
        logger = pl.loggers.TensorBoardLogger(logpath, name=name,version=None)
        if is_cluster():
            progress_bar_refresh_rate=0
        else:
            progress_bar_refresh_rate=10

        trainer = pl.Trainer(gpus=1, max_epochs=60, callbacks=[ckpt_callback], logger=logger, fast_dev_run=self.debug, 
            gradient_clip_val=5., benchmark=True, progress_bar_refresh_rate=progress_bar_refresh_rate)

        train_dataloader = Loader_in["Train"]
        if "Valin" in Loader_in.keys():
            val_dataloader = Loader_in["Valin"]
            test_dataloader = Loader_in["Val"]
        else:
            val_dataloader = Loader_in["Val"]
            test_dataloader = Loader_in["Test"]

        trainer.fit(model=self.model,train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
        trainer.test(test_dataloaders=test_dataloader)
        self.train_path = self.model.logger.experiment.log_dir

    def get_score(self, X, *args, **kwargs):
        data_loader =  DataLoader(TensorDataset(torch.Tensor(X)), num_workers=12, batch_size=256)

        scores = []
        self.model.freeze()
        with torch.no_grad():
            for i, x in enumerate(data_loader):
                x = x[0]
                x = x.to(self.model.device)
                score =self.model.get_scores(x)
                scores.append(score.detach())
        scores = torch.cat(scores, dim=0).to('cpu').numpy()
        self.model.unfreeze()
        return scores

    # def predict(self, X):
    #     scores = self.get_score(X)
    #     # scores, _, _ = normal(scores, min=self.min, max=self.max)
    #     prediction = scores>= self.threshold
    #     return prediction
    
    # def set_normal(self, X):
    #     scores = self.get_score(X)
    #     self.min = np.min(scores)
    #     self.max = np.max(scores)

    # def set_threshold(self, X, Y):
    #     self.set_normal(X)
    #     scores = self.get_score(X)
    #     scores_normed, _, _ = normal(scores, min=self.min, max=self.max)
    #     _, self.threshold = find_best_val(scores_normed, Y, get_acc, max_steps=200)
    #     self.threshold = self.threshold*(self.max-self.min)+self.min

    # def setup(self, X, Y, mode='Val',*args, **kwargs):
    #     self.set_normal(X[mode])
    #     self.set_threshold(X[mode], Y[mode])

    def save_model(self, save_path, filename = 'finalized_model.sav'):
        if self.train_path is not None:
            model_path = os.path.join(save_path, filename)
            torch.save(self.model.state_dict(), model_path)
            with open(os.path.join(save_path, "model_paths.yaml"), 'w+') as file:
                yaml.dump({"inn":save_path}, file)

    def load_model(self, save_path, filename='finalized_model.sav'):
        model_path = os.path.join(save_path, filename)
        # state_dict = self.model.state_dict()
        if os.path.isfile(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.model.device)
                self.model.load_state_dict(state_dict)
            except Exception as exc:
                raise Exception("Unexpected Exception occured while loading: \n{}".format(model_path)) from exc 
            return self
        else:
            print("No saved Model at path: {}".format(model_path))
            return self
        