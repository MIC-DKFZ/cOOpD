import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from algo.ano_encoder.feature_likelihood import OOD_Encoder
from algo.model import load_best_model
from algo.utils import get_seg_im_gt
from config.datasets.brain import get_brain_args
from datamodules.brain_module import get_brain_dataset
from datamodules.brain import get_brain_dataset_eval
from metrics.binary_score import get_metrics
from pyutils.ano_algo import visualize_gt, visualize_segmentations, visualize_scores
from pyutils.dictof import DictOf
from latent_gen.ood_model import Abstract_OOD
from tqdm import tqdm as tqdm
from pyutils.py import is_cluster
from config.latent_model import filename, model_dicts, rel_save
from argparse import ArgumentParser

from collections import defaultdict
import pandas as pd
import csv
from sys import getsizeof

parser = ArgumentParser()
parser.add_argument('-p', '--path', type=str, default= '/home/silvia/Documents/CRADL/logs_cradl/pretext/brain/simclr-VGG/default/version_83')
parser.add_argument('--name_exp', type=str, default='full')


def evaluate_ano_loader(ano_algo, dataloader, kwargs_pix=dict(), kwargs_sample=dict(), scoring=['sample', 'pixel']):
    """Evaluates the anomaly detection algorithm on the given dataloader.
    Returns all batch information, all sample scores and all pixel scores.
    When not all information is interesting use skipy_keys to leave this info.

    Args:
        experiment ([type]): [description]
        dataloader ([type]): [description]
        skip_keys (list, optional): [description]. Defaults to [].
        kwargs_pix ([type], optional): [description]. Defaults to dict().
        kwargs_sample ([type], optional): [description]. Defaults to dict().

    Returns:
        [type]: [description]
    """
    ano_algo.freeze()
    batch_dict = DictOf()
    batch_dict.skip_keys = ('data')
    score_sample_dict = DictOf()
    score_pixel_dict = DictOf()

    if is_cluster() is False:
        dataloader = tqdm(dataloader)
    
    for i, batch in enumerate(dataloader):
        #batch['labels'], batch['seg'] = get_seg_im_gt(batch)

        batch_dict.insert(batch)

        # x = batch['data']
        # x = x.to(ano_algo.device)

        if 'sample' in scoring:
            score_sample = ano_algo.score_samples(batch, **kwargs_sample)
            score_sample_dict.insert(score_sample)

        if 'pixel' in scoring:
            score_pix = ano_algo.score_pixels(batch, **kwargs_pix)
            score_pixel_dict.insert(score_pix)

    ano_algo.zero_grad()
    torch.cuda.empty_cache()

    #scores to patient level
    #score_sample_dict.insert(batch_dict['patient_name'])

    return batch_dict, score_sample_dict, score_pixel_dict 



def get_pixel_metrics(batch_dict, score_pixel_dict):
    """Computes metrics over all pixels

    Args:
        batch_dict ([type]): [description]
        score_pixel_dict ([type]): [description]

    Returns:
        [type]: [description]
    """
    seg = batch_dict['seg']
    out_dict = dict()

    times=[time.time()]
    for key, val in score_pixel_dict.items():
        out_dict[key] = dict()
        assert seg.shape == val.shape
        val_ = np.reshape(val, -1)
        seg_ = np.reshape(seg, -1)
        out_dict[key] = get_metrics(val_, seg_)
        times.append(time.time())
    times= [times[i+1]-times[i] for i in range(len(times)-1)]
    meantime = np.mean(np.array(times))
    print("Mean Time for Metric Computation: {}".format(meantime))
    return out_dict

def get_brain_metrics(batch_dict, score_pixel_dict):
    """Computes metrics over the brain area

    Args:
        batch_dict ([type]): [description]
        score_pixel_dict ([type]): [description]

    Returns:
        [type]: [description]
    """
    seg = batch_dict['seg']
    out_dict = dict()
    brain_region = batch_dict['data']!=0
    seg_ = seg[brain_region]
    seg_ = np.reshape(seg_, -1)
    times=[time.time()]
    for key, val in score_pixel_dict.items():
        out_dict[key] = dict()
        val_ = val[brain_region]
        val_ = np.reshape(val_, -1)
        assert seg_.shape == val_.shape
        out_dict[key] = get_metrics(val_, seg_)
        times.append(time.time())
    times= [times[i+1]-times[i] for i in range(len(times)-1)]
    meantime = np.mean(np.array(times))
    print("Mean Time for Eval: {}".format(meantime))
    return out_dict

#change this to get metrics per patient_name
def get_sample_metrics(batch_dict, score_sample_dict):
    labels = batch_dict['label'].flatten()
    patient_names = batch_dict['patient_name'].flatten()

    names_labels_dict = dict(zip(patient_names, labels))
    #labels without repeated values
    labels = np.array(list(names_labels_dict.values()))


    out_dict = dict()
    for key, val in score_sample_dict.items():
        out_dict[key + 'mean', key + 'median', key + 'Q1'] = dict()

        score_names_sample_dict = {}
        print(val)
        print(patient_names)

        for key_n, value_n in zip(patient_names, val):
            if key_n not in score_names_sample_dict:
                score_names_sample_dict[key_n] = [value_n]
            else:
                score_names_sample_dict[key_n].append(value_n)

        print(score_names_sample_dict)
        print(score_names_sample_dict.values())
        print(score_names_sample_dict.keys())
        print(score_names_sample_dict.items())

        dataframe1 = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in score_names_sample_dict.items()]))
        print(dataframe1)
        print(patient_names)

        for k in score_names_sample_dict:
            print(k)
            print(score_names_sample_dict[k])
            print(sum(score_names_sample_dict[k]) / len(score_names_sample_dict[k]))
            print(np.mean(score_names_sample_dict[k]))
            print(np.median(score_names_sample_dict[k]))
            print(np.quantile(score_names_sample_dict[k], 0.25))
            print(min(score_names_sample_dict[k]))

        avg_score_names_sample_dict = {k: np.mean(score_names_sample_dict[k]) for k in score_names_sample_dict}
        #To get a score (AUROC etc) by image, uncomment:
        val_avg = np.array(list(avg_score_names_sample_dict.values()))

        median_score_names_sample_dict = {k: np.median(score_names_sample_dict[k]) for k in score_names_sample_dict}
        #To get a score (AUROC etc) by image, uncomment:
        val_median = np.array(list(median_score_names_sample_dict.values()))

        Q1_score_names_sample_dict = {k: np.quantile(score_names_sample_dict[k], 0.25) for k in score_names_sample_dict}
        #To get a score (AUROC etc) by image, uncomment:
        val_Q1 = np.array(list(Q1_score_names_sample_dict.values()))


        assert labels.shape == val_avg.shape
        out_dict[key + 'mean'] = get_metrics(val_avg, labels)
        out_dict[key + 'median'] = get_metrics(val_median, labels)
        out_dict[key + 'Q1'] = get_metrics(val_Q1, labels)
    return out_dict
        

def eval_encoder_model(path, Model_cls=Abstract_OOD, version='results_plot_1', model_kwargs={'n_components':1},
            name_exp='isles_brats'):
    # Load and Prepare Model
    ano_algo, model_name = get_ood_encoder(path, Model_cls=Model_cls, model_kwargs=model_kwargs, version=version, get_name=True)
    ano_algo = ano_algo.to('cuda:0')
    model_path = os.path.join(path, version, model_name)
    
    # Load and Prepare Data #change
    exp_loader_dict = get_data_loaders(name_exp)

    # Evaluation
    # pixel_dict, sample_dict, scan_dict = eval_ano_algo(ano_algo, model_path, exp_loader_dict, name_exp=name_exp)
    # return pixel_dict, sample_dict, scan_dict, model_name
    pixel_dict, sample_dict = eval_ano_algo(ano_algo, model_path, exp_loader_dict, name_exp=name_exp)
    return pixel_dict, sample_dict, model_name


def eval_ano_algo(ano_algo, model_path, loader_dict, scoring=['sample'],vis=True, name_exp='fast'):
    """Function which evalutes the dataloader on the pre-specified datasets (based on name_exp)
    Iterates over evaluate_ano_laoder()
    Saves results in the model_path (pixel, 2d, 3d)
    Returns a dict with metrics for pixels, samples & scans

    Args:
        ano_algo ([type]): [description]
        model_path ([type]): [description]
        loader_dict ([type]): [description]
        vis (bool, optional): [description]. Defaults to True.
        name_exp (str, optional): [description]. Defaults to 'fast'.

    Returns:
        [type]: [description]
    """
    sample_dict = dict()
    pixel_dict = dict()
    data_dict = dict()
    for dataset, loader in loader_dict.items():
        print(dataset)
        print(loader)
        loader_pixels = dict()
        for name, kw_px in kwargs_eval.items(): #this part was wrong in the original code from Carsten
            if 'pixel' in scoring:
                print("Compute Scores over Dataset")
                batch_dict, score_sample_dict, score_pixel_dict = evaluate_ano_loader(ano_algo=ano_algo,
                                                                                      dataloader=loader,
                                                                                      kwargs_pix=kw_px,
                                                                                      scoring=['sample', 'pixel'])
                print("Compute Pixel Wise Metrics")
                pixel_metrics = get_pixel_metrics(batch_dict, score_pixel_dict)
                loader_pixels[name] = pixel_metrics
                save_path = os.path.join(model_path, dataset, name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if vis:
                    visualize_segmentations(batch_dict, score_sample_dict, score_pixel_dict, save_path, name=dataset)
            else:
                batch_dict, score_sample_dict, score_pixel_dict = evaluate_ano_loader(ano_algo=ano_algo, dataloader=loader, kwargs_pix=kw_px, scoring=scoring)
        save_path = os.path.join(model_path, dataset)
        batch_score_dict = {**batch_dict, **score_sample_dict}
        keys_to_remove = ['metadata_origin', 'metadata_spacing', 'metadata_direction']
        for key in keys_to_remove:
            del batch_score_dict[key]

        df = pd.DataFrame({k: list(v) for k, v in batch_score_dict.items()})
        df.rename(columns={"nll": 'score'})



        visualize_scores(score_sample_dict, batch_dict, save_path, name=dataset)
        if vis and 'pixel' in scoring:
            visualize_gt(model_path, batch_dict, dataset)

        sample_metrics = get_sample_metrics(batch_dict, score_sample_dict)
        sample_dict[dataset] = sample_metrics
        pixel_dict[dataset] = loader_pixels

        data_dict[dataset] = dict()
        # data_dict[dataset]['Num Slices'] = int(batch_dict['seg'].shape[0])
        # data_dict[dataset]['Num  Slices Positive'] = int(batch_dict['labels'].sum())
        # data_dict[dataset]['Prevalence Slices'] = float(batch_dict['labels'].sum()/batch_dict['seg'].shape[0])
        # data_dict[dataset]['Prevalence Voxels'] = float(batch_dict['seg'].sum()/batch_dict['seg'].size)
        # data_dict[dataset]['Num Voxels'] = int(batch_dict['seg'].size)
        # data_dict[dataset]['Num Voxels Positive'] = int(batch_dict['seg'].sum())

        df.to_csv(os.path.join(save_path, 'all_info.csv'))

    with open(os.path.join(model_path, 'pixel_{}.yaml'.format(name_exp)), 'w+') as file:
        yaml.dump(pixel_dict, file)
    with open(os.path.join(model_path, 'samples_{}.yaml'.format(name_exp)), 'w+') as file:
        yaml.dump(sample_dict, file)
    with open(os.path.join(model_path, 'data_{}.yaml'.format(name_exp)), 'w+') as file:
        yaml.dump(data_dict, file)
    # with open(os.path.join(model_path, 'data_info_score_{}.csv'.format(name_exp)), 'w+') as file:
    #     yaml.dump(data_dict, file)
    return pixel_dict, sample_dict




##################### SET PARAMETERS ############################
    

version= rel_save

def get_data_loaders(name_exp):
    from config.datasets.brain import eval_loader_args
    arg_data = get_brain_args(mode='eval', data_type='default', cond=False)
    if name_exp == 'fast':
        keys = ['hcp_syn_fast']
    elif name_exp == 'full':
        keys = [key for key in eval_loader_args.keys()]
    elif name_exp == 'val':
        keys = ['hcp_syn_val', 'isles_val', 'brats_val']
        
    exp_loader_dict = dict()
    for key in keys:
        exp_loader_dict[key] = get_brain_dataset_eval(**arg_data['common_args'], **eval_loader_args[key])
    
    return exp_loader_dict

kwargs_eval = {
    'eps_0':{
        'eps': 0.0,
        'n_runs':1
        },
}

##################### End SET PARAMETERS ############################

def main(path, version='results_plot_0', name_exp='fast'):
    pixel_dicts = {}
    sample_dicts = {}
    for model_dict in model_dicts:
        # print (model_dict
        pixel_dict, sample_dict, model_name = eval_encoder_model(path=path, version=version, name_exp=name_exp, **model_dict)
        pixel_dicts[model_name] = pixel_dict
        sample_dicts[model_name] = sample_dict
    with open(os.path.join(path, version, f"version_pixel_{name_exp}.yaml"), 'w') as file:
        yaml.dump(pixel_dicts, file)
    with open(os.path.join(path, version, f"version_sample_{name_exp}.yaml"), 'w') as file:
        yaml.dump(sample_dicts)
    return pixel_dicts, sample_dicts
    

        


def get_ood_encoder(path, Model_cls=Abstract_OOD, version='results_plot_0', model_kwargs={'n_components':1}, get_name=False):
    encoder, args = load_best_model(path)
    model_kwargs.update({'n_features':args.z_dim})
    model_kwargs.update({'input_shape':args.z_dim})
    if get_name:
        gen_model, model_name = get_model(path, Model_cls=Model_cls, model_kwargs=model_kwargs, version=version, get_name=True)
    else:
        gen_model = get_model(path, Model_cls=Model_cls, model_kwargs=model_kwargs, version=version)

    ood_encoder=  OOD_Encoder(encoder, gen_model)
  
    if get_name:
        return ood_encoder, model_name
    return ood_encoder


def get_model(loading_path, Model_cls=Abstract_OOD ,version='results_plot_0', model_kwargs={'z_dim':512, 'n_components':1}, get_path=False, get_name=False):
    model = Model_cls(**model_kwargs)
    model_path = os.path.join(loading_path, version, model.name)
    model = model.load_model(model_path, filename=filename)
    if get_path:
        return model.model, model_path
    elif get_name:
        return model.model, model.name
    return model.model






if __name__ == "__main__":
    args = parser.parse_args()
    path = args.path 
    name_exp = args.name_exp
    main(path, version=version, name_exp=name_exp)
