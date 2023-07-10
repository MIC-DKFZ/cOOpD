import torch 
import os
import json
import numpy as np
# import pytorch_lightning as pl
import torch.distributed as dist
import torch.nn.functional as F

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
    #print(x)
    #print(y)
    return x, y




def process_batch_for_nnclr(batch):
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
        loc = torch.from_numpy(np.array(batch['metadata']['location'], dtype=np.int)).to(device=x[0].device)
        pat_num = batch['patch_num']
        pat_name = torch.from_numpy(np.asarray([int(i.replace('copdgene_', '')[1:]) for i in batch['patient_name']]))
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
    #print(x)
    #print(y)
    return x, loc, pat_num, pat_name




# TODO: Find a way to make this easily generalizable! (multi channel etc.)
#might be changing the appearance
def norm(x, val_range=(0,1), in_range=(-1.5, 1.5)):
    out =  x - in_range[0]  #x.min()
    out /=  (in_range[1] - in_range[0]) # x.max()
    if val_range != (0,1):
        out *= (val_range[1]-val_range[0])
        out += val_range[0]
    return out

######################### ALRIGHT from here ############################

def get_seg_im_gt(batch):
    seg = batch["seg"].float() > 0.5
    img_gt = (torch.sum(seg, dim=(1, 2, 3)) >= 1).to(dtype=torch.long).detach() #.tolist()
    pixel_gt = seg.detach() #.tolist()
    return img_gt, pixel_gt

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


class GatherLayer(torch.autograd.Function):
    """Gathers tensors from all processes, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
            dist.all_gather(output, input)
        else:
            output = [input]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        if dist.is_available() and dist.is_initialized():
            grad_out = torch.zeros_like(input)
            grad_out[:] = grads[dist.get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def gather(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)
def nnclr_loss_func(nn: torch.Tensor, p: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Computes NNCLR's loss given batch of nearest-neighbors nn from view 1 and
    predicted features p from view 2.

    Args:
        nn (torch.Tensor): NxD Tensor containing nearest neighbors' features from view 1.
        p (torch.Tensor): NxD Tensor containing predicted features from view 2
        temperature (float, optional): temperature of the softmax in the contrastive loss. Defaults
            to 0.1.

    Returns:
        torch.Tensor: NNCLR loss.
    """

    nn = F.normalize(nn, dim=-1)
    p = F.normalize(p, dim=-1)

    logits = nn @ p.T / temperature

    n = p.size(0)
    labels = torch.arange(n, device=p.device)

    loss = F.cross_entropy(logits, labels)
    return loss
