import logging

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state
from tabpfn import TabPFNClassifier

from pycilt.automl.automl_core import AutomlClassifier


class AutoTabPFNClassifier(AutomlClassifier):
    def __init__(self, n_ensembles=100, device='cpu', random_state=None):
        self.logger = logging.getLogger(name=AutoTabPFNClassifier.__name__)
        self.random_state = check_random_state(random_state)
        self.device = device
        self.n_ensembles = n_ensembles
        self.model = None

    def fit(self, X, y, **kwd):
        self.model = TabPFNClassifier(device=self.device, N_ensemble_configurations=self.n_ensembles)
        self.model.fit(X, y, overwrite_warning=True)

    def predict(self, X, verbose=0):
        y_pred = self.model.predict(X, return_winning_probability=False, normalize_with_test=False)
        return y_pred

    def score(self, X, y, sample_weight=None, verbose=0):
        y_pred = self.model.predict(X, return_winning_probability=False, normalize_with_test=False)
        acc = accuracy_score(y, y_pred)
        print(acc)
        return acc

    def predict_proba(self, X, verbose=0):
        y_pred = self.model.predict_proba(X, normalize_with_test=True, return_logits=False)
        return y_pred

    def decision_function(self, X, verbose=0):
        y_pred = self.model.predict_proba(X, normalize_with_test=True, return_logits=False)
        return y_pred
