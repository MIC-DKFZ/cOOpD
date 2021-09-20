import pytorch_lightning as pl
from algo.utils import process_batch
from pyutils.gradients import get_image_gradient
from data_aug.square_noise import smooth_tensor, MedianPool2d

import torch

class OOD_Encoder(pl.LightningModule):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder= encoder
        self.head = head
        
    def score_samples(self, batch):
        x, y = process_batch(batch)
        x = x.to(self.device)
        scores = self._score_samples(x).detach().cpu()
        return {'nll':scores}
    
    def _score_samples(self, x):
        return self.head.get_scores(self.encoder.encode(x))
    
    def score_pixels(self, batch, grad_type='vanilla', eps=0, n_runs=2):
        x, y = process_batch(batch)
        # mask = x!= 0
        x = x.to(self.device)
        mask = x!= 0
        def err_fn(x):
            loss = self._score_samples(x).sum()
            return loss

        loss = get_image_gradient(model=self, inpt=x, err_fn=err_fn, smooth=True, n_runs=n_runs, eps=eps,
                                 grad_type=grad_type)


        score_dict = {
            'nll' : loss,
        } 
        score_dict = postprocess_nll(score_dict, mask=mask, score='nll')
        for key in score_dict:
            score_dict[key] = score_dict[key].detach().cpu()

        return score_dict


pool_tensor = MedianPool2d(kernel_size=5, same=True)

def postprocess_nll(batch_dict:dict, mask, score='loss'):
    with torch.no_grad():
        mask_keys = [score]
        def mask_(tensor):
            return mask_tensor(tensor, mask)
        batch_dict = add_datadict(batch_dict, mask_, mask_keys, 'mask')
        # keys_sm = [key for key in batch_dict.keys() if key.split('_')[0] in mask_keys]
        # batch_dict = add_datadict(batch_dict, smooth_tensor, keys_sm, 'sm8')
        # pool_tensor = MedianPool2d(kernel_size=5, same=True)

        # keys_p = [key for key in batch_dict.keys() if key.split('_')[0] in mask_keys]
        keys_p = [score+'_mask']
        batch_dict = add_datadict(batch_dict, pool_tensor, keys_p, 'p5')
        # keys_sm2 = [score+'_mask_p5', score+'_mask_p5', score]
        keys_sm2 = [score+'_mask_p5']
        batch_dict = add_datadict(batch_dict, smooth_tensor, keys_sm2, 'sm8')

        for key in keys_p+keys_sm2 :
            _ = batch_dict.pop(key)
    
        return batch_dict



def mask_tensor(tensor, mask):
    return tensor*mask


def add_datadict(batch_dict, function, keys, add_name):
    for key in keys:
        name = key+"_{}".format(add_name)
        batch_dict[name] = function(batch_dict[key])
    return batch_dict