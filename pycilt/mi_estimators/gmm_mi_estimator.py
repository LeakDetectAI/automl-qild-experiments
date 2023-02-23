import copy
import logging

import numpy as np
from infoselect import get_gmm, SelectVars
from sklearn.linear_model import LogisticRegression

from .mi_base_class import MIEstimatorBase
from ..utils import log_exception_error


class GMMMIEstimator(MIEstimatorBase):
    def __init__(self, n_classes, input_dim, y_cat=False, reg_covar=1e-06, random_state=42):
        super().__init__(n_classes=n_classes, input_dim=input_dim, random_state=random_state)
        self.y_cat = y_cat
        self.num_comps = list(np.arange(2, 20, 1))
        self.reg_covar = reg_covar
        self.n_models = 10

        # Classification Model
        self.cls_model = None
        self.best_model = None
        self.best_seed = None
        self.round = None
        self.logger = logging.getLogger(GMMMIEstimator.__name__)

    def get_goodnessof_fit(self, gmm, X, y):
        if isinstance(gmm, dict):
            classes = list(set(y))
            bic = []
            likelihood = []
            n_components = []
            for c in classes:
                bic.append(gmm[c].bic(X[y == c]))
                likelihood.append(gmm[c].score(X[y == c]))
                n_components.append(gmm[c].n_components)
            bic = np.mean(bic)
            likelihood = np.mean(likelihood)
            n_components = np.mean(n_components)
        else:
            Z = np.hstack((y.reshape((-1, 1)), X))
            bic = gmm.bic(Z)
            likelihood = gmm.score(Z)
            n_components = gmm.n_components
        return bic, likelihood, n_components

    def fit(self, X, y, verbose=0, **kwd):
        bics_hats = []
        best_likelihood = -np.inf
        seed = self.random_state.randint(2 ** 31, dtype="uint32")
        for iter_ in range(self.n_models):
            self.logger.info(f"++++++++++++++++++ GMM Model {iter_} ++++++++++++++++++")
            try:
                gmm = get_gmm(X, y, y_cat=self.y_cat, num_comps=self.num_comps, reg_covar=self.reg_covar,
                              val_size=0.33, random_state=seed + iter_)
                select = SelectVars(gmm, selection_mode='backward')
                select.fit(X, y, verbose=verbose, eps=np.finfo(np.float32).eps)
                mi_mean, _ = select.get_info().values[0][1], select.get_info().values[0][2]
                mi = np.max([mi_mean, 0.0]) * np.log2(np.e)
                if not (np.isnan(mi) or np.isinf(mi)):
                    bic, likelihood, n_components = self.get_goodnessof_fit(gmm, X, y)
                    self.logger.info(f"MI {np.around(mi, 4)}  BIC {np.around(bic, 4)} Likelihood {np.around(likelihood, 4)} n_components {n_components}")
                    bics_hats.append([bic, mi])
                    if best_likelihood < likelihood:
                        self.logger.info(f"GMM Model {iter_} set best with likelihood {np.around(likelihood, 4)}")
                        best_likelihood = likelihood
                        self.best_model = copy.deepcopy(select)
                        self.best_seed = seed + iter_
                else:
                    self.logger.info(f"Model {iter_} trained estimates wrong MI")
            except Exception as error:
                log_exception_error(self.logger, error)
                self.logger.error(f"Model {iter_} was not valid ")
        self.create_classification_model(X, y)
        return self

    def create_classification_model(self, X, y, **kwd):
        if self.best_model is not None:
            idx = np.where(self.best_model.get_info()['delta'].values < 0)
            try:
                self.logger.info(self.best_model.get_info())
                self.logger.info(f"Indices {idx[0]}")
                self.round = idx[0][0] - 1
            except IndexError as error:
                log_exception_error(self.logger, error)
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
        try:
            bic, likelihood, n_components = self.get_goodnessof_fit(self.best_model.gmm, X, y)
            mi_mean, _ = self.best_model.get_info().values[0][1], self.best_model.get_info().values[0][2]
            mi = np.max([mi_mean, 0.0]) * np.log2(np.e)
            self.logger.debug(f"MI {np.around(mi, 4)}  BIC {np.around(bic, 4)} Likelihood {np.around(likelihood, 4)} n_components {n_components}")
            score = likelihood
            self.logger.debug(f"Best Model is not None out of {self.n_models} score {score}")
        except Exception as error:
            self.logger.debug("Best Model is None")
            log_exception_error(self.logger, error)
            score = 0.0
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
        iter_ = 0
        while True:
            iter_ += 1
            gmm = get_gmm(X, y, y_cat=self.y_cat, num_comps=self.num_comps, reg_covar=self.reg_covar,
                          val_size=0.33, random_state=self.best_seed)
            select = SelectVars(gmm, selection_mode='backward')
            select.fit(X, y, verbose=verbose, eps=np.finfo(np.float32).eps)
            mi_mean, _ = select.get_info().values[0][1], select.get_info().values[0][2]
            mi_estimated = np.max([mi_mean, 0.0]) * np.log2(np.e)
            if verbose:
                print(f'Model Number: {iter_}, Estimated MI: {mi_estimated}')
            self.logger.info(f'Model Number: {iter_}, Estimated MI: {mi_estimated}')
            if np.isnan(mi_estimated) or np.isinf(mi_estimated):
                self.logger.error(f'Nan MI Re-estimating')
            else:
                break
            if iter_ > 100:
                if np.isnan(mi_estimated) or np.isinf(mi_estimated):
                    self.logger.error(f'Setting Mi to 0')
                    mi_estimated = 0.0
                break
        return mi_estimated
