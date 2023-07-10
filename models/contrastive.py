import torch
import torch.nn as nn
from models.base import MLP, _prediction_mlp, _get_nnclr_projection_head

class SimCLR(nn.Module):
    def __init__(self, encoder, num_ft, dim_out, non_lin=nn.ReLU, batch_norm=False):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.projection = MLP(dim_in=num_ft, dim_out=dim_out, hidden_dims=[num_ft], non_lin=non_lin, bn=batch_norm)

    def encode(self, x, **kwargs):
        return self.encoder(x)
    
    def forward(self, x):
        # TODO: This needs to be doublecked for VAE since the model will have a different encoder etc...
        h = self.encoder(x)
        #pteste = torch.flatten(h, start_dim=1)
        z = self.projection(torch.flatten(h, start_dim=1))
        return z, h


class NNCLR(nn.Module):
    def __init__(self, encoder, num_ft, dim_out, non_lin=nn.ReLU, batch_norm=False):
        super(NNCLR, self).__init__()
        self.encoder = encoder
        self.projection = MLP(dim_in=num_ft, dim_out=dim_out, hidden_dims=[num_ft], non_lin=non_lin, bn=batch_norm)

        #self.projection_head = _get_nnclr_projection_head(num_ftrs = num_ft, h_dims= [num_ft], out_dim = dim_out, num_layers = 3)

        self.prediction_head = _prediction_mlp(in_dims = num_ft, h_dims = 2048, out_dims = dim_out)
    def encode(self, x, **kwargs):
        return self.encoder(x)

    def forward(self, x):
        h = self.encoder(x)
        # pteste = torch.flatten(h, start_dim=1)
        z = self.projection(torch.flatten(h, start_dim=1))
        p = self.prediction_head(z)
        return z, h, p