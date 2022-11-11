import copy
import logging

import numpy as np
from infoselect import get_gmm, SelectVars
from sklearn.linear_model import LogisticRegression

from .mi_base_class import MIEstimatorBase


class GMMMIEstimator(MIEstimatorBase):
    def __init__(self, n_classes, input_dim, y_cat=False, max_num_components=0, reg_covar=1e-06, random_state=42):
        super().__init__(n_classes=n_classes, input_dim=input_dim, random_state=random_state)
        self.y_cat = y_cat
        self.max_num_components = max_num_components
        if max_num_components == 0:
            self.num_comps = [2, 5, 10, 15, 20]
        else:
            self.num_comps = list(np.arange(2, self.max_num_components, 3))
        self.reg_covar = reg_covar
        self.models = None
        self.cls_model = None
        # Classification Model
        self.best_model = None
        self.best_model_idx = None
        self.round = None
        self.logger = logging.getLogger(GMMMIEstimator.__name__)

    def fit(self, X, y, verbose=0, **kwd):
        self.models = []
        for i, val_size in enumerate(np.linspace(0.20, 0.80, num=30)):
            try:
                gmm = get_gmm(X, y, y_cat=self.y_cat, num_comps=self.num_comps, val_size=val_size,
                              reg_covar=self.reg_covar, random_state=self.random_state)
                select = SelectVars(gmm, selection_mode='backward')
                select.fit(X, y, verbose=verbose, eps=np.finfo(np.float32).eps)
                self.models.append(select)
                self.logger.info(f"Model {i} trained with validation data {val_size.round(2)}")
            except Exception as e:
                self.logger.info(f"Model {i} was not valid {val_size.round(2)}")

        gmm = get_gmm(X, y, random_state=self.random_state)
        select = SelectVars(gmm, selection_mode='backward')
        select.fit(X, y, verbose=verbose, eps=np.finfo(np.float32).eps)
        self.models.append(select)
        self.logger.info(f"Default Model trained with validation data {0.33}")
        self.create_best_model(X, y, verbose=verbose, **kwd)
        return self

    def create_best_model(self, X, y, verbose=0, **kwd):
        self.estimate_mi(X, y, verbose=verbose, create_best_model=True, **kwd)
        if self.best_model_idx is not None:
            self.best_model = copy.deepcopy(self.models[self.best_model_idx])
            idx = np.where(self.best_model.get_info()['delta'].values < 0)
            try:
                self.logger.info(self.best_model.get_info())
                self.logger.info(f"Indexses {idx}")
                self.round = idx[0][0] - 1
            except IndexError as e:
                self.round = 0
            X_new = self.best_model.transform(X, rd=self.round)
            self.cls_model = LogisticRegression()
            self.cls_model.fit(X_new, y)
        else:
            self.cls_model = LogisticRegression()
            self.cls_model.fit(X, y)

    def predict(self, X, verbose=0):
        if self.best_model is not None:
            X = self.best_model.transform(X, rd=self.round)
        return self.cls_model.predict(X=X)

    def score(self, X, y, sample_weight=None, verbose=0):
        return self.estimate_mi(X, y, verbose=verbose)

    def predict_proba(self, X, verbose=0):
        if self.best_model is not None:
            X = self.best_model.transform(X, rd=self.round)
        return self.cls_model.predict_proba(X=X)

    def decision_function(self, X, verbose=0):
        if self.best_model is not None:
            X = self.best_model.transform(X, rd=self.round)
        return self.cls_model.decision_function(X=X)

    def estimate_mi(self, X, y, verbose=0, create_best_model=False, **kwd):
        mi_hats = []
        for iter_, model in enumerate(self.models):
            iterations = 100
            while iterations > 0:
                model.fit(X, y, verbose=verbose, eps=np.finfo(np.float32).eps)
                mi_mean, _ = model.get_info().values[0][1], model.get_info().values[0][2]
                mi_mean = np.abs(mi_mean)
                iterations -= 1
                if not (np.isnan(mi_mean) or np.isinf(mi_mean)):
                    break
            if verbose:
                print(f'Iter: {iter_}, Estimated MI: {mi_mean}')
            self.logger.info(f'Iter: {iter_}, Estimated MI: {mi_mean}')
            mi_hats.append(mi_mean)

        mi_hats = np.array(mi_hats)
        if len(mi_hats) > 0:
            n = int(len(self.models) / 3)
            mi_hats = mi_hats[np.argpartition(mi_hats, -n)[-n:]]

        mi_estimated = np.nanmean(mi_hats)
        if np.isnan(mi_estimated) or np.isinf(mi_estimated):
            self.logger.error(f'Setting MI to 0')
            mi_estimated = 0

        if create_best_model:
            if len(mi_hats) > 0:
                self.best_model_idx = np.argmax(mi_hats)
                self.logger.info(f'Best model: {self.best_model_idx}, Estimated MI: {mi_hats[self.best_model_idx]}')
            else:
                self.best_model_idx = None
        return mi_estimated
