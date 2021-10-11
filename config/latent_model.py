from latent_gen.ood_model import GaussianMixtureOOD
from latent_gen.inn import INN_latent


filename = 'finalized_model.sav'
tmp = 'latent_tmp'
suffix = '_data'
rel_save='results_plot_0'

model_dicts = [
    {'Model_cls': GaussianMixtureOOD, 'model_kwargs':{'n_components':1}},
    {'Model_cls': GaussianMixtureOOD, 'model_kwargs':{'n_components':2}},
    {'Model_cls': GaussianMixtureOOD, 'model_kwargs':{'n_components':4}},
    {'Model_cls': GaussianMixtureOOD, 'model_kwargs':{'n_components':8}},
    {'Model_cls': INN_latent, 'model_kwargs':{}}
]