import copy
import gc

import torch

from .sklearn_leakage_detector import SklearnLeakageDetector
from ..automl import AutoTabPFNClassifier
from ..bayes_search import BayesSearchCV
from ..bayes_search_utils import get_scores, log_callback, update_params_at_k
from ..constants import *
from ..utils import log_exception_error


class TabPFNLeakageDetector(SklearnLeakageDetector):
    def __init__(self, padding_name, learner_params, fit_params, hash_value, cv_iterations, n_hypothesis,
                 base_directory, search_space, hp_iters, n_inner_folds, validation_loss, random_state=None, **kwargs):
        super().__init__(padding_name=padding_name, learner_params=learner_params, fit_params=fit_params,
                         hash_value=hash_value, cv_iterations=cv_iterations, n_hypothesis=n_hypothesis,
                         base_directory=base_directory, search_space=search_space, hp_iters=hp_iters,
                         n_inner_folds=n_inner_folds, validation_loss=validation_loss, random_state=random_state,
                         **kwargs)
        self.n_jobs = 8
        self.base_detector = AutoTabPFNClassifier

    def perform_hyperparameter_optimization(self, X, y):
        X_train, y_train = self.get_training_dataset(X, y)
        learner = self.base_detector(**self.learner_params)
        bayes_search_params = dict(estimator=learner, search_spaces=self.search_space, n_iter=self.hp_iters,
                                   scoring=self.validation_loss, n_jobs=self.n_jobs, cv=self.inner_cv_iterator,
                                   error_score=0, random_state=self.random_state,
                                   optimizers_file_path=self.optimizers_file_path)
        bayes_search = BayesSearchCV(**bayes_search_params)
        search_keys = list(self.search_space.keys())
        search_keys.sort()
        self.logger.info(f"Search Keys {search_keys}")
        callback = log_callback(search_keys)
        try:
            bayes_search.fit(X_train, y_train, groups=None, callback=callback, **self.fit_params)
        except Exception as error:
            log_exception_error(self.logger, error)
            self.logger.error("Cannot fit the Bayes SearchCV ")
        train_size = X_train.shape[0]

        if learner is not None:
            del learner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return train_size, bayes_search, search_keys

    def fit(self, X, y):
        if self._is_fitted_:
            self.logger.info(f"Model already fitted for the padding {self.padding_code}")
            self.results[RANDOM_CLASSIFIER][ACCURACY] = []
            for k, (train_index, test_index) in enumerate(self.cv_iterator.split(X, y)):
                self.logger.info(f"********************************* Split {k + 1} *********************************")
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                self.calculate_random_classifier_accuracy(X_train, y_train, X_test, y_test)
            self.store_results_random()
        else:
            train_size, bayes_search, search_keys = self.perform_hyperparameter_optimization(X, y)
            for k, (train_index, test_index) in enumerate(self.cv_iterator.split(X, y)):
                self.logger.info(f"********************************* Split {k + 1} *********************************")
                train_index = train_index[:train_size]
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                self.calculate_random_classifier_accuracy(X_train, y_train, X_test, y_test)
                self.calculate_majority_voting_accuracy(X_train, y_train, X_test, y_test)
            self.store_results()
            for i in range(self.n_hypothesis):
                learner_params = copy.deepcopy(self.learner_params)
                loss, learner_params = update_params_at_k(bayes_search, search_keys, learner_params, k=i)
                self.logger.info(f"*************  Model {i + 1} with loss {loss} and parameters {learner_params} ********************")
                model = self.base_detector(**learner_params)
                for k, (train_index, test_index) in enumerate(self.cv_iterator.split(X, y)):
                    train_index = train_index[:train_size]
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    model.fit(X=X_train, y=y_train)
                    p_pred, y_pred = get_scores(X_test, model)
                    self.evaluate_scores(X_test, X_train, y_test, y_train, y_pred, p_pred, model, i)
            self.store_results()


