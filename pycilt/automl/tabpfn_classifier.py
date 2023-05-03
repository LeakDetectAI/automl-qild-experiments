import logging

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state
from tabpfn import TabPFNClassifier


class AutoTabPFNClassifier(BaseEstimator, ClassifierMixin):
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
        y_pred = self.model.predict(X)
        return y_pred

    def score(self, X, y, sample_weight=None, verbose=0):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def predict_proba(self, X, verbose=0):
        y_pred = self.model.predict_proba(X, normalize_with_test=True, return_logits=False)
        return y_pred

    def decision_function(self, X, verbose=0):
        y_pred = self.model.predict_proba(X, normalize_with_test=True, return_logits=False)
        return y_pred

    def get_params(self, deep=True):
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
