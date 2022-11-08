import logging

import numpy as np
from infoselect import get_gmm, SelectVars
from sklearn.linear_model import LogisticRegression

from .mi_base_class import MIEstimatorBase


class GMMMIEstimator(MIEstimatorBase):
    def __init__(self, n_classes, input_dim, y_cat=False, max_num_components=30, reg_covar=1e-06, random_state=42):
        super().__init__(n_classes=n_classes, input_dim=input_dim, random_state=random_state)
        self.y_cat = y_cat
        self.max_num_components = max_num_components
        self.num_comps = np.arange(2, self.max_num_components)
        self.reg_covar = reg_covar
        self.models = None
        self.best_model = None
        self.cls_model = None
        self.logger = logging.getLogger(MIEstimatorBase.__name__)

    def fit(self, X, y, verbose=0, **kwd):
        self.models = []
        for val_size in np.linspace(0.20, 0.80, num=20):
            gmm = get_gmm(X, y, y_cat=self.y_cat, num_comps=self.num_comps, val_size=val_size,
                          reg_covar=self.reg_covar, random_state=self.random_state)
            select = SelectVars(gmm, selection_mode='backward')
            self.models.append(select)
        self.create_best_model(X, y, verbose=verbose, **kwd)
        return self

    def create_best_model(self, X, y, verbose=0, **kwd):
        self.estimate_mi(X, y, verbose=verbose, **kwd)
        idx = np.where(self.best_model.get_info()['delta'].values < 0)
        rd = idx[0][0] - 1
        X_new = self.best_model.transform(X, rd=rd)
        self.cls_model = LogisticRegression()
        self.cls_model.fit(X_new, y)

    def predict(self, X, verbose=0):
        return self.cls_model.predict(X=X, verbose=verbose)

    def score(self, X, y, sample_weight=None, verbose=0):
        return self.estimate_mi(X, y, verbose=verbose)

    def predict_proba(self, X, verbose=0):
        return self.cls_model.predict_proba(X=X, verbose=verbose)

    def decision_function(self, X, verbose=0):
        return self.cls_model.decision_function(X=X, verbose=verbose)

    def estimate_mi(self, X, y, verbose=0, **kwd):
        mi_hats = []
        for model in self.models:
            model.fit(X, y, verbose=verbose, eps=np.finfo(np.float32).eps)
            mi_mean, _ = model.get_info().values[0][1], model.get_info().values[0][2]
            mi_hats.append(mi_mean)
        mi_hat = np.mean(mi_hats)
        self.best_model = self.models[np.argmax(mi_hats)]
        return mi_hat
