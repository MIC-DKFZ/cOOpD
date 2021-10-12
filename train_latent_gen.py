import os 
from argparse import ArgumentParser

import numpy as np

from pyutils.parser import str2bool
from latent_gen.ood_model import Abstract_OOD
from datamodules.brain_module import BrainDataModule
from algo.model import load_best_model, get_label_latent
from config.latent_model import filename, model_dicts, tmp, suffix, rel_save

parser = ArgumentParser()
parser.add_argument('-p', '--path', type=str)
parser.add_argument('--num_epoch', type=int, default=1)
parser.add_argument('--resave', type=str2bool, nargs='?', const=True, default=False)


def fit_model(model:Abstract_OOD, X, Y, save_path, mode='Train'):
    model.fit(X, Y)
    model.setup(X, Y, mode=mode)
    model.save_model(save_path, filename=filename)
    return model

def init_model(model_dict, n_feat, path, rel_save=None):
    if rel_save is not None:
        rel_save = os.path.join(path, rel_save)
        if not os.path.exists(rel_save):
            os.makedirs(rel_save)
    model_dict['model_kwargs'].update(dict(n_features=n_feat, input_shape=n_feat, path=path))
    model = model_dict['Model_cls'](**model_dict['model_kwargs'])
    save_path = os.path.join(rel_save, model.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return model, save_path

def save_outputs(path, resave=False, num_epoch=1, get_slice_idx=True):
    tmp_dir = os.path.join(path, tmp)
    suffix = '_data'
    keys = ['Train', 'Valin']
    if not (all([os.path.exists(os.path.join(tmp_dir, key.lower()+suffix+'.npz')) for key in keys]) and resave is False):
        experiment, args = load_best_model(path)
        experiment = experiment.to('cuda:0')
        mode='val'
        if num_epoch != 1:
            mode = 'train'
        datamodule = BrainDataModule(mode=mode)

        loader_dict = {'train': datamodule.train_dataloader(), 'val':datamodule.val_dataloader()}
        
        data_dict = dict()
        # from pdb import set_trace as bp 
        # bp()
        for key1, key2 in zip(keys, ['train', 'val']):
            if key1 =='Train':
                data_dict[key1]= get_label_latent(experiment, loader_dict[key2], get_slice=get_slice_idx, num_epoch=num_epoch)
            else:
                data_dict[key1]= get_label_latent(experiment, loader_dict[key2], get_slice=get_slice_idx)
            
        if os.path.exists(tmp_dir) is False:
            os.mkdir(tmp_dir)
        for key in data_dict.keys():
            np.savez_compressed(os.path.join(tmp_dir, key.lower()+suffix), **data_dict[key])
            
def load_tmp_data(path, mode='val', get_slice=False):
    loading_path = os.path.join(path, tmp, mode+suffix+'.npz')
    loaded = np.load(loading_path)
    
    if get_slice:
        return loaded["latent"], loaded['labels'], loaded['slice_idxs']
    return loaded["latent"], loaded['labels'], None

def load_outputs(path, c=0):
    X, Y = dict(), dict()
    X['Train'], Y['Train'], _ = load_tmp_data(path, mode='train')
    X['Valin'], Y['Valin'], _ = load_tmp_data(path, mode='valin')
    if c is not None:
        for k in Y.keys():
            Y[k] = get_ood_labels(Y[k],c=c)
    return X, Y


def get_ood_labels(y, c=0):
    mask = y==c
    y[mask] = 0
    y[~mask] = 1
    return y


def main(path, resave=False, num_epoch=1):
    save_outputs(path, get_slice_idx=False, resave=resave, num_epoch=num_epoch)
    X, Y = load_outputs(path)
    n_feat = X['Train'].shape[1]
    for model_dict in model_dicts:
        model, save_path = init_model(model_dict, n_feat, path, rel_save=rel_save)
        fit_model(model, X, Y, save_path)


if __name__ == "__main__":
    args = parser.parse_args()
    path = args.path 
    resave =  args.resave 
    num_epoch = args.num_epoch
    main(path, resave=resave, num_epoch=num_epoch)