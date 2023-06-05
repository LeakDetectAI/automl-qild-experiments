import logging

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import check_random_state
from tabpfn import TabPFNClassifier

from pycilt.automl.automl_core import AutomlClassifier
from pycilt.automl.reduction_techniques_tabpfn import reduction_techniques, n_reduced


class AutoTabPFNClassifier(AutomlClassifier):
    def __init__(self, n_features, n_classes, n_ensembles=100, reduction_technique='select_from_model_rf',
                 random_state=None, **kwargs):
        self.n_features = n_features
        self.n_classes = n_classes
        self.logger = logging.getLogger(name=AutoTabPFNClassifier.__name__)
        self.random_state = check_random_state(random_state)
        if reduction_technique not in reduction_techniques.keys():
            raise ValueError(f"Reduction type {reduction_technique} not defined {reduction_techniques.keys()}")
        self.reduction_technique = reduction_technique
        self.selection_model = reduction_techniques[reduction_technique]
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = device
        self.logger.info(f"Device {self.device}")
        self.n_ensembles = n_ensembles
        self.__is_fitted__ = False
        self.model = None

    def transform(self, X, y=None):
        if not self.__is_fitted__:
            if self.n_features != X.shape[-1]:
                raise ValueError(f"Dataset passed does not contain {self.n_features}")
            if self.n_classes != len(np.unique(y)):
                raise ValueError(f"Dataset passed does not contain {self.n_classes}")
            self.logger.info(f"Fitting the reduction model to reduce the {self.n_features} to {n_reduced}")
            self.selection_model.fit(X, y)
            self.__is_fitted__ = True
        if self.n_features > 100:
            X = self.selection_model.transform(X)
        return X

    def fit(self, X, y, **kwd):
        X = self.transform(X, y)
        self.model = TabPFNClassifier(device=self.device, N_ensemble_configurations=self.n_ensembles)
        self.model.fit(X, y, overwrite_warning=True)

    def predict(self, X, verbose=0):
        p = self.predict_proba(X, verbose=0)
        y_pred = np.argmax(p, axis=-1)
        self.logger.info("Predict Done")
        return y_pred

    def score(self, X, y, sample_weight=None, verbose=0):
        y_pred = self.predict(X)
        acc = balanced_accuracy_score(y, y_pred)
        return acc

    def predict_proba(self, X, verbose=0):
        X = self.transform(X)
        self.logger.info("Predict_proba Transform Done")
        y_pred = self.model.predict_proba(X, normalize_with_test=True, return_logits=False)
        self.logger.info("Predict_proba Done")
        return y_pred

    def decision_function(self, X, verbose=0):
        return self.predict_proba(X, verbose)
