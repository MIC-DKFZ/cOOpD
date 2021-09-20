import os
from config.paths import glob_conf
from copy import deepcopy

data_dir = glob_conf['datapath']
tmp_data_dir = None


datasets_common_args = {
        'batch_size': 64,
        'target_size': 128,
        'input_slice': 1,
        'add_noise': True,
        'mask_type': 'test', #'test',  # 0.0, ## TODO: Fix this for later!
        'elastic_deform': False,
        'rnd_crop': True,
        'rotate': True,
        'color_augment': True,
        'add_slices': 0,
        'double_headed' : False,
        'base_train' : 'default'
    }

datasets_train_args = {
        'base_dir': [data_dir + '/brain/hcp_train/'],
        'slice_offset': 20,
        'num_processes': 12,
        'tmp_dir': tmp_data_dir,
    }

datasets_val_args = {
        'base_dir': [data_dir + '/brain/hcp_eval/'],
        'n_items': 6400,
        'do_reshuffle': False,
        'mode': 'val',
        'num_processes': 4,
        'slice_offset': 20,
        'tmp_dir': tmp_data_dir,
    }

datasets_val_ano_args = {
        "base_dir": [ data_dir + "/brain/hcp_syn_ano/from_imgs_default/"],
        "n_items": 6400,
        "do_reshuffle": False,
        "mode": "val",
        "num_processes": 4,
        "slice_offset": 10,
        "label_slice": 2,
        "tmp_dir": tmp_data_dir,
    }

datasets_test_args = {
        "base_dir": [ data_dir + "/brain/hcp_syn_ano/from_imgs/"],
        "n_items": 6400,
        "do_reshuffle": False,
        "mode": "val",
        "num_processes": 4,
        "slice_offset": 10,
        "label_slice": 2,
        "tmp_dir": tmp_data_dir,
    }

eval_val_ano_args = {
        'base_dir':[data_dir+ '/brain/hcp_syn_ano/from_imgs_val/'],
        'n_items': None,
        'do_reshuffle': False,
        'mode': 'val',
        'num_processes': 4,
        'slice_offset': 10,
        'label_slice': 2,
        'tmp_dir': tmp_data_dir,
    }

eval_test_args = {
        'base_dir':[data_dir+ '/brain/hcp_syn_ano/from_imgs_test/'], 
        'n_items': None,
        'do_reshuffle': False,
        'mode': 'val',
        'num_processes': 4,
        'slice_offset': 10,
        'label_slice': 2,
        'tmp_dir': tmp_data_dir,
    }


eval_loader_args = {
    'hcp_syn_fast' : {
        'base_dir': data_dir + '/brain/hcp_syn_ano/from_imgs_val' ,
        'n_items' : 1000,
        'do_reshuffle': False,
        'mode': 'val',
        'num_processes': 8,
        'slice_offset': 10,
        'label_slice': 2,
        'tmp_dir': tmp_data_dir,
    },
    'hcp_syn_val' : {
        'base_dir': data_dir + '/brain/hcp_syn_ano/from_imgs_val' ,
        'n_items' : None,
        'do_reshuffle': False,
        'mode': 'val',
        'num_processes': 8,
        'slice_offset': 10,
        'label_slice': 2,
        'tmp_dir': tmp_data_dir,
    },
    'hcp_syn_test' : {
        'base_dir': data_dir + '/brain/hcp_syn_ano/from_imgs_test' ,
        'n_items' : None,
        'do_reshuffle': False,
        'mode': 'val',
        'num_processes': 8,
        'slice_offset': 10,
        'label_slice': 2,
        'tmp_dir': tmp_data_dir,
    },
    'brats' : {
        'base_dir': data_dir + '/brain/brats17_test' ,
        'n_items' : None,
        'do_reshuffle': False,
        'mode': 'val',
        'num_processes': 8,
        'slice_offset': 10,
        'label_slice': 2,
        'tmp_dir': tmp_data_dir,
    },
    'isles' : {
        'base_dir': data_dir + '/brain/isles15_siss_test',
        'n_items' : None,
        'do_reshuffle': False,
        'mode': 'val',
        'num_processes': 8,
        'slice_offset': 10,
        'label_slice': 2,
        'tmp_dir': tmp_data_dir,
    },
    'isles_val':{
        'base_dir': data_dir + '/brain/isles15_siss',
        'n_items' : None,
        'do_reshuffle': False,
        'mode': 'val',
        'num_processes': 8,
        'slice_offset': 10,
        'label_slice': 2,
        'tmp_dir': tmp_data_dir,
    },
    'brats_val':{
        'base_dir' : data_dir + '/brain/brats17',
        'n_items' : None,
        'do_reshuffle': False,
        'mode': 'val',
        'num_processes': 8,
        'slice_offset': 10,
        'label_slice': 2,
        'tmp_dir': tmp_data_dir,
    }
}

train_eval = {
      'mode' : 'val'
}

def get_brain_args(mode='eval', data_type='default', cond=False):
    """Returns the default arguments to generate the brain dataset, 
    given the mode and the type of datasets wished.

    Args:
        mode (str, optional): [description]. Defaults to 'eval'.
        type (str, optional): [description]. Defaults to 'default'.

    Raises:
        NotImplementedError: [description]

    Returns:
        [type]: [description]
    """
    d_common = deepcopy(datasets_common_args)
    d_train = deepcopy(datasets_train_args)
    d_val = deepcopy(datasets_val_args)
    d_val_ano = deepcopy(eval_val_ano_args)
    d_test = deepcopy(eval_test_args)
    if mode=='eval':


        d_train.update(train_eval)
        if cond is True:
            cond_update={'slice_offset': 10}
            d_train.update(cond_update)
            d_val.update(cond_update)
            
    elif mode == 'train':
        pass
        # return d_common, d_train, d_val, d_val_ano, d_test
    else:
        raise NotImplementedError
    args_dict = {
                'common_args':d_common,
                'trainset_args': d_train, 
                'valset_args': d_val, 
                'valanoset_args': d_val_ano, 
                'testset_args': d_test}
    return args_dict



        

    