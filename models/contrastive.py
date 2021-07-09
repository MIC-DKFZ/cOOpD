import torch
import torch.nn as nn
from models.base import MLP

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
        z = self.projection(torch.flatten(h, start_dim=1))
        return z, h