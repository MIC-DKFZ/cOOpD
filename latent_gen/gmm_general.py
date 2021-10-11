import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

# Based on Implemetation from David Zimmerer

class GaussModel(torch.nn.Module):
    def __init__(self, mu, cov, reg_cov=0.001):
        super(GaussModel, self).__init__()

        self.mu = torch.nn.Parameter(mu)
        self._cov = torch.nn.Parameter(cov)

        self.d = self.mu.shape[0]
        self.reg_cov = reg_cov
        self.init = False

    def _init_distr(self):
        self.distribution = MultivariateNormal(self.mu, self._cov)

    def to(self, device, *args, **kwargs):
        super().to(device=device, *args, **kwargs)
        self.init=False    

    def log_prob(self, X, differentiable=False, k_means=False):

        if not k_means:
            if differentiable:
                cov_ = self._cov @ self._cov.transpose(0, 1) + torch.eye(self.d) * self.reg_cov
                # cov_ = torch.eye(self.d)
            else:
                if not self.init:
                    self._init_distr()

                return self.distribution.log_prob(X)
        else:
            # Running k-means algorithm
            cov_ = torch.eye(self.d).to(self._cov.device)
            log_p = MultivariateNormal(self.mu, cov_).log_prob(X)
            return log_p

    def cov(self):
        return self._cov

    def sample(self, differentiable=False):
        if differentiable:
            cov_ = self._cov @ self._cov.transpose(0, 1) + torch.eye(self.d) * self.reg_cov
        else:
            cov_ = self._cov
        return MultivariateNormal(self.mu, cov_).sample()

def random_GM(n_dims, min=0, max=1, device='cpu'):
    mu = (torch.rand(n_dims, device=device) * (max - min)) + (min)
    cov = torch.eye(n_dims, device=device)

    return GaussModel(mu, cov)

class GMM(torch.nn.Module):
    def __init__(self, n_components, n_dims, init_min=0, init_max=1, reg_cov=1e-5, k_means=False):
        super(GMM, self).__init__()

        self.n_components = n_components

        self.pis = torch.nn.Parameter(
            torch.ones(self.n_components) / self.n_components
        )
        self.k_means= k_means

        self.gms = torch.nn.ModuleList(
            random_GM(n_dims, init_min, init_max, device='cpu') for _ in range(self.n_components)
        )

        self.register_buffer('reg_cov', torch.eye(n_dims) * reg_cov)

        shape = [1, n_dims]
        self.register_buffer('in_means', torch.zeros(size=shape))
        self.register_buffer('in_stds', torch.ones(size=shape))

    

    def init_preprocessing(self, in_means:torch.Tensor, in_stds:torch.Tensor, eps=1e-5):
        assert(in_means.shape == self.in_means.shape) # Shape of means is not equal
        assert(in_stds.shape == self.in_stds.shape) # Shape of stds is not equal
    
        self.in_means[:] = in_means #.to(self.device) 
        self.in_stds[:] = in_stds+ eps #.to(self.device) + eps



    def e_step(self, x, k_means=False):
        with torch.no_grad():
            x =  (x-self.in_means)/self.in_stds
            ws = torch.zeros(x.shape[0], self.n_components, device=x.device)
            ws_log = torch.zeros(x.shape[0], self.n_components, device=x.device)
            for i, gm in enumerate(self.gms):
                log_probs = gm.log_prob(x, k_means=k_means)

                ws_log[:, i] = log_probs + torch.log(self.pis[i])
                
            ws_log_c = F.softmax(ws_log, dim=1)
        return ws_log_c

    def m_step(self, x, ws, k_means=False):
        with torch.no_grad():
            x =  (x-self.in_means)/self.in_stds
            for i, gm in enumerate(self.gms):
                pi_new_unormalized = torch.sum(ws[:, i])
                pi_new = pi_new_unormalized / x.shape[0]
                mu_new =  torch.sum((ws[:, i, None]/pi_new_unormalized) *x , dim=0) 

                x_m  = (x-mu_new[None, :]) * ws[:, i, None]

                if k_means is False:
                    cov_new = (x_m.transpose(1,0) @ x_m )/pi_new_unormalized
                    cov_new += torch.eye(x.shape[1], device=x.device) * self.reg_cov  
                else:
                    cov_new = torch.eye(x.shape[1], device=x.device)

                gm.mu.data = mu_new
                gm._cov.data = cov_new
                self.pis[i] = pi_new

    def em_step(self, x, k_means=False):
        ws = self.e_step(x, k_means=k_means)
        self.m_step(x, ws, k_means=k_means)

    def log_prob(self, x, to_mean=True, differentiable=False, k_means=False):
        x =  (x-self.in_means)/self.in_stds
        if not k_means:
            if differentiable:
                pis_ = F.softmax(self.pis, dim=0)
            else:
                pis_ = self.pis
        else:
            pis_ = torch.ones(self.n_components) / self.n_components
            pis_.to(x.device)

        ws = torch.zeros(x.shape[0], self.n_components, device=x.device)
        for i, gm in enumerate(self.gms):
            log_probs = gm.log_prob(x, differentiable=differentiable, k_means=k_means)
            ws[:, i] = log_probs + torch.log(pis_[i])

        log_p = torch.logsumexp(ws, dim=1)
        if to_mean:
            log_p = torch.mean(log_p)

        return log_p

    def get_scores(self, x):
        return -1* self.score_samples(x)

    def score_samples(self, x):
        return  self.log_prob(x, to_mean=False)

    def predict(self, x, differentiable=False):
        x =  (x-self.in_means)/self.in_stds
        if differentiable:
            pis_ = F.softmax(self.pis, dim=0)
        else:
            pis_ = self.pis

        ws = torch.zeros(x.shape[0], self.n_components, device=x.device)
        for i, gm in enumerate(self.gms):
            log_probs = gm.log_prob(x, differentiable=differentiable)
            ws[:, i] = torch.exp(log_probs) * pis_[i]

        return torch.argmax(ws, dim=1)

    def fit(self, x, convergence_limit=1e-2, max_iter=1000):
        in_means = x.mean(dim=0, keepdim=True)
        in_stds=x.std(dim=0, keepdim=True)
        self.init_preprocessing(in_means=in_means, in_stds=in_stds)


        self._fit(x, convergence_limit, max_iter, k_means=True)
        if self.k_means is False:
            self._fit(x, convergence_limit, max_iter)

    def _fit(self, x, convergence_limit, max_iter, k_means=False):
        log_prob = self.log_prob(x, k_means=k_means)
        with torch.no_grad():
            for i in range(max_iter):
                self.em_step(x, k_means=k_means)
                log_prob_new = self.log_prob(x, k_means=k_means)
                if abs(log_prob_new - log_prob) < convergence_limit:
                    break
                else:
                    log_prob = log_prob_new


    def sample(self, n_samples=1, differentiable=False):
        samples = []
        for i in range(n_samples):
            pi_idx = np.random.choice(list(range(self.n_components)), p=F.softmax(self.pis, dim=0).detach().numpy())
            sample = self.gms[pi_idx].sample(differentiable=differentiable)

            samples.append(sample)
        samples =  torch.stack(samples)
        samples =  samples * self.in_std + self.in_means 
        return samples