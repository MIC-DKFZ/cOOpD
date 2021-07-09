import torch 
import os
import json
import numpy as np
# import pytorch_lightning as pl

###################### POSSIBLY CHANGES NEEDED! #########################
def process_batch(batch):
    """Preprocesses the output of a dataloader (batch) to predictor(x) and response (y).
    Batch is either a tuple (x, y) or a dictionary with 'data/ ('label' or 'seg' or None) 

    Args:
        batch ([type]): [description]

    Raises:
        Exception: [description]
        Exception: [description]

    Returns:
        [type]: [description]
    """
    if isinstance(batch, dict):
        x = batch['data']
        if 'seg' in batch.keys():
            y = batch['seg']
            # y = batch["seg"].float() > 0.5
        elif 'label' in batch.keys():
            y = batch['label']
        else:
            if isinstance(x, (list, tuple)):
                y = torch.zeros(x[0].shape[0], device=x[0].device, dtype=torch.int)
            elif torch.is_tensor(x):
                y = torch.zeros(x.shape[0], device=x.device, dtype=torch.int)
            else:
                raise Exception("Unknown type as input")

    elif isinstance(batch, (tuple, list)):
        x, y = batch
    else:
        raise Exception("Batch Type {} is not implemented".format(type(batch)))
    if torch.is_tensor(x):# TODO: possibly slows down training - look for alternative
        if not torch.is_floating_point(x):
            x = x.to(torch.float)
        else:
            x = x.to(torch.float)
    if isinstance(x, (tuple, list)):
        # print("DATA IS TUPLE")
        x_ = []
        for data in x:
            if torch.is_tensor(data):# TODO: possibly slows down training - look for alternative
                x_.append(data.to(torch.float))
        x = tuple(x_)
        del x_
    # print(x)
    return x, y

# TODO: Find a way to make this easily generalizable! (multi channel etc.)
def norm(x, val_range=(0,1), in_range=(-1.5, 1.5)):
    out =  x - in_range[0]  #x.min()
    out /=  (in_range[1] - in_range[0]) # x.max()
    if val_range != (0,1):
        out *= (val_range[1]-val_range[0])
        out += val_range[0]
    return out

######################### ALRIGHT from here ############################


def exclude_from_wt_decay(named_params, weight_decay, skip_list=['bias', 'bn']):
    params = []
    excluded_params = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        elif any(layer_name in name for layer_name in skip_list):
            excluded_params.append(param)
        else:
            params.append(param)
    return[
        {'params': params, 'weight_decay':weight_decay},
        {'params': excluded_params, 'weight_decay':0}
    ]


