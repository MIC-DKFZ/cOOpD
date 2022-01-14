from abc import abstractmethod
import os
import sklearn
from sklearn.svm import SVC
from sklearn import metrics, mixture
import numpy as np
import matplotlib.pyplot as plt
import torch
import joblib

# from mixture.mahalanobis import sample_estimator_single_class, Mahalanobis
# from validation.utils import normal, histogram_scores, barplot_scores_vs_slice, histogram_scores_slice
# from metrics.binary_score import find_best_val, get_acc
# from validation.gmm import GaussianMixture
from latent_gen.gmm_general import GMM



class Abstract_OOD():
    name='Abstract_OOD'
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X:dict, Y:dict, *args, **kwargs):
        pass

    @abstractmethod
    def get_score(self, X:np.array, *args, **kwargs):
        """Returns the scores for the given data X

        Args:
            X (np.array): [NxD]
        
        Returns:
            scores (np.array) : [N] 
        """
        pass

    # @abstractmethod
    # def predict(self, X:np.array):
    #     """Returns Predictions for the given data X

    #     Args:
    #         X (np.array): [NxD]
        
    #     Returns:
    #         scores (np.array) : [N] 
    #     """
    #     pass

    def setup(self, *args, **kwargs):
        pass

    # def get_acc(self, X:np.array, Y:np.array, *args, **kwargs):
    #     prediction = self.predict(X)
    #     acc= float(np.sum(prediction==Y))/len(Y)
    #     return acc


    def save_model(self, save_path, filename = 'finalized_model.sav'):
        joblib.dump(self, os.path.join(save_path, filename))

    def load_model(self, save_path, filename='finalized_model.sav'):
        model_path = os.path.join(save_path, filename)
        if os.path.isfile(model_path):
            try:
                model = joblib.load(model_path)
            except Exception as exc:
                raise Exception("Unexpected Exception Occured while loading: \n{}".format(model_path)) from exc 
            return model
        else:
            print("No saved Model at path: {}".format(model_path))
            return self

class GaussianMixtureOOD(Abstract_OOD):
    name = "GMM"
    def __init__(self, n_components=1, n_features = 512, *args, **kwargs):
        super().__init__()
        self.model = GMM(n_dims=n_features, n_components=n_components)
        self.name = self.name+" {} Comp".format(n_components)
        self.threshold=None
        self.device = 'cuda:0'
        # self.device = 'cpu'
        self.model = self.model.to(self.device)

    def _fit(self, X:np.ndarray, Y:np.ndarray, c=0):
        print(X.shape, Y.shape)
        X_tr = X[Y.flatten()==c]
        X_tr = torch.Tensor(X_tr).to(self.device)
        self.model.fit(X_tr)

    def fit(self, X:dict, Y:dict, c=0, *args, **kwargs):
        self._fit(X['Train'], Y["Train"], c=c)

    def get_score(self, X:np.ndarray, *args, **kwargs):
        X_ = torch.Tensor(X).to(self.device)
        return - self.model.score_samples(X_).detach().to('cpu').numpy()


    # def setup(self, X, Y, mode='Val',*args, **kwargs):
    #     self.set_threshold(X[mode], Y[mode])