import copy
import logging

import numpy as np
from infoselect import get_gmm, SelectVars
from sklearn.linear_model import LogisticRegression

from .mi_base_class import MIEstimatorBase
from ..utils import log_exception_error


class GMMMIEstimator(MIEstimatorBase):
    def __init__(self, n_classes, input_dim, n_models=10, y_cat=False, reg_covar=1e-06,
                 random_state=42):
        super().__init__(n_classes=n_classes, input_dim=input_dim, random_state=random_state)
        self.y_cat = y_cat
        self.n_models = n_models
        self.num_comps = list(np.arange(2, 20, 3))
        self.reg_covar = reg_covar
        self.models = None

        # Classification Model
        self.cls_model = None
        self.best_model = None
        self.best_mi = None
        self.round = None
        self.logger = logging.getLogger(GMMMIEstimator.__name__)

    def fit(self, X, y, verbose=0, **kwd):
        self.models = []
        val_sizes = list(np.linspace(0.20, 0.80, num=self.n_models)) + [0.33]
        mi_hats = []
        self.best_mi = -np.nan
        for iter_, val_size in enumerate(val_sizes):
            val = np.around(val_size, 2)
            try:
                gmm = get_gmm(X, y, y_cat=self.y_cat, num_comps=self.num_comps, val_size=val_size,
                              reg_covar=self.reg_covar, random_state=self.random_state)
                select = SelectVars(gmm, selection_mode='backward')
                gmm_model = copy.deepcopy(select)
                select.fit(X, y, verbose=verbose, eps=np.finfo(np.float32).eps)
                mi_mean, _ = select.get_info().values[0][1], select.get_info().values[0][2]
                mi = np.abs(np.max([mi_mean, 0.0]) * np.log2(np.e))
                if not (np.isnan(mi) or np.isinf(mi)):
                    mi_hats.append(mi)
                    self.models.append(gmm_model)
                    self.logger.info(f"Model {iter_} trained with validation data {val} with mi {mi}")
                    if mi > self.best_mi:
                        self.logger.info(f"Setting Model {iter_} with val size {val} "
                                         f"as best model with mi {mi}")
                        self.best_mi = mi
                        self.best_model = copy.deepcopy(select)
                else:
                    self.logger.info(f"Model {iter_} trained with validation data {val} estimates wrong MI")
            except Exception as error:
                log_exception_error(self.logger, error)
                self.logger.error(f"Model {iter_} was not valid {val}")
        return self

    def create_best_model(self, X, y, verbose=0, **kwd):
        if self.best_model is not None:
            idx = np.where(self.best_model.get_info()['delta'].values < 0)
            try:
                self.logger.info(self.best_model.get_info())
                self.logger.info(f"Indices {idx}")
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
        if self.best_model is not None:
            X = self.best_model.transform(X, rd=self.round)
            score = self.cls_model.score(X=X, y=y)
            self.logger.info(f"Best Model is not None total {len(self.models)}")
        else:
            score = 0.0
            self.logger.info("Best Model is None")
        return score

    def predict_proba(self, X, verbose=0):
        if self.best_model is not None:
            X = self.best_model.transform(X, rd=self.round)
        return self.cls_model.predict_proba(X=X)

    def decision_function(self, X, verbose=0):
        if self.best_model is not None:
            X = self.best_model.transform(X, rd=self.round)
        return self.cls_model.decision_function(X=X)

    def estimate_mi(self, X, y, verbose=0, **kwd):
        mi_hats = np.zeros(len(self.models))
        for iter_, model in enumerate(self.models):
            lmodel = copy.deepcopy(model)
            lmodel.fit(X, y, verbose=verbose, eps=np.finfo(np.float32).eps)
            mi_mean, _ = lmodel.get_info().values[0][1], lmodel.get_info().values[0][2]
            mi_hats[iter_] = np.abs(np.max([mi_mean, 0.0]) * np.log2(np.e))
            if verbose:
                print(f'Model Number: {iter_}, Estimated MI: {mi_hats[iter_]}')
            self.logger.info(f'Model Number: {iter_}, Estimated MI: {mi_hats[iter_]}')

        mi_estimated = np.nanmean(mi_hats)
        if np.isnan(mi_estimated) or np.isinf(mi_estimated):
            self.logger.error(f'Setting MI to 0')
            mi_estimated = 0
        return mi_estimated
