from latent_gen.ood_model import GaussianMixtureOOD
from latent_gen.inn import INN_latent
from latent_gen import attention_mech
from latent_gen import cnn_aftercradl


filename = 'finalized_model.sav'
tmp = 'latent_tmp'
suffix = '_data'
rel_save='results_plot_0'

model_dicts = [ #deleted GMMs
    # {'Model_cls': GaussianMixtureOOD, 'model_kwargs':{'n_components':1}},
    # {'Model_cls': GaussianMixtureOOD, 'model_kwargs':{'n_components':2}},
    # {'Model_cls': GaussianMixtureOOD, 'model_kwargs':{'n_components':4}},
    {'Model_cls': GaussianMixtureOOD, 'model_kwargs':{'n_components':8}},
    # {'Model_cls': INN_latent, 'model_kwargs':{}}
    # {'Model_cls': attention_mech, 'model_kwargs': {}},
    # {'Model_cls': cnn_aftercradl, 'model_kwargs': {}}
]