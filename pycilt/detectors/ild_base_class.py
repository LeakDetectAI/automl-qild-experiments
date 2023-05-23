import logging
import os
from abc import ABCMeta

import h5py
import numpy as np
from scipy.stats import fisher_exact
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils import check_random_state
from statsmodels.stats.multitest import multipletests

from pycilt.bayes_search_utils import get_scores
from pycilt.classifiers import MajorityVoting
from pycilt.contants import *
from pycilt.detectors.utils import leakage_detection_methods, mi_estimation_metrics
from pycilt.statistical_tests import paired_ttest
from pycilt.utils import log_exception_error, create_directory_safely


class InformationLeakageDetector(metaclass=ABCMeta):
    def __int__(self, padding_name: str, learner_params: dict, fit_params: dict, hash_value: str, cv_iterations: int, n_hypothesis: int,
                base_directory: str, random_state: object, **kwargs):
        self.logger = logging.getLogger(InformationLeakageDetector.__name__)
        self.padding_name = padding_name
        self.fit_params = fit_params
        self.learner_params = learner_params
        self.cv_iterations = cv_iterations
        self.n_hypothesis = n_hypothesis

        self.hash_value = hash_value
        self.random_state = check_random_state(random_state)
        self.cv_iterator = StratifiedKFold(n_splits=self.cv_iterations, random_state=random_state, shuffle=True)

        self.estimators = []
        self.results = {}
        self.base_detector = None
        self.base_directory = base_directory
        self.optimizers_file_path = os.path.join(self.base_directory, OPTIMIZER_FOLDER, f"{hash_value}.pkl")
        self.results_file = os.path.join(self.base_directory, RESULT_FOLDER, f"{hash_value}_intermediate.h5")
        create_directory_safely(self.results_file, True)
        create_directory_safely(self.optimizers_file_path, True)
        self.__initialize_objects__()

    @property
    def padding_name(self):
        self.padding_name = '_'.join(self.padding_name.split(' ')).lower()
        self.padding_name = self.padding_name.replace(" ", "")
        return self.padding_name

    @padding_name.setter
    def padding_name(self, value):
        self._padding_name = value

    @property
    def _is_fitted_(self) -> bool:
        if os.path.exists(self.results_file):
            file = h5py.File(self.results_file, 'r')
            conditions = []
            if self.padding_name in file:
                self.logger.info(f"Simulations done for padding label {self.padding_name}")
                conditions.append(self.padding_name in file)
                for model_name in self.results.keys():
                    padding_name_group = file[self.padding_name]
                    conditions.append(model_name in padding_name_group)
                    self.logger.info(f"Predictions done for model {model_name}")
        return np.all(conditions)


    def __initialize_objects__(self):
        for i in range(self.n_hypothesis):
            self.results[f'model_{i}'] = {}
            for metric_name, evaluation_metric in mi_estimation_metrics.items():
                self.results[f'model_{i}'][metric_name] = []
        self.results[MAJORITY_VOTING][ACCURACY] = []

    def get_training_dataset(self, X, y):
        lengths = []
        for i, (train_index, test_index) in enumerate(self.cv_iterator.split(X, y)):
            lengths.append(len(train_index))
        test_size = X.shape[0] - np.min(lengths)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)
        train_index, test_index = list(sss.split(X, y))[0]
        return X[train_index], y[train_index]

    def calculate_majority_voting_accuracy(self, X_train, y_train, X_test, y_test):
        estimator = MajorityVoting()
        estimator.fit(X_train, y_train)
        p_pred, y_pred = get_scores(X_test, estimator)
        accuracy = accuracy_score(y_test, y_pred)
        self.results[MAJORITY_VOTING][ACCURACY].append(accuracy)
        self.logger.info(f"Majority Voting Performance Metric {ACCURACY}: Value {accuracy}")

    def perform_hyperparameter_optimization(self, X, y):
        raise NotImplemented

    def fit(self, X, y):
        raise NotImplemented

    def store_results(self):
        self.logger.info(f"Result file {self.results_file}")
        if os.path.exists(self.results_file):
            file = h5py.File(self.results_file, 'a')
        else:
            file = h5py.File(self.results_file, 'w')
        padding_name_group = file.create_group(self.padding_name)
        for model_name, metric_results in self.results.items():
            model_group = padding_name_group.create_group(model_name)
            for metric_name, results in metric_results.items():
                model_group.create_dataset(metric_name, results)
        file.close()

    def read_results_file(self, detection_method):
        metric_name = leakage_detection_methods[detection_method]
        model_results = {}
        if os.path.exists(self.results_file):
            file = h5py.File(self.results_file, 'r')
            padding_name_group = file[self.padding_name]
            for model_name in self.results.keys():
                model_group = padding_name_group[model_name]
                try:
                    model_results[model_name] = model_group[metric_name]
                except KeyError as e:
                    log_exception_error(self.logger, e)
                    self.logger.error("Error while calibrating the probabilities")
                    model_results = None
                    raise ValueError(f"Provided Metric Name {metric_name} is not applicable "
                                     f"for current base detector {self.base_detector} "
                                     f"so cannot apply the provided detection method {detection_method}")
            file.close()
            return model_results
        else:
            raise ValueError(f"The results are not found at the path {self.results_file}")

    def detect(self, detection_method):
        # change for including holm-bonnfernoi
        def holm_bonferroni(p_values):
            reject, pvals_corrected, _, alpha = multipletests(p_values, 0.01, method='holm', is_sorted=False)
            reject = [False] * len(p_values) + list(reject)
            pvals_corrected = [1.0] * len(p_values) + list(pvals_corrected)
            return p_values, pvals_corrected, reject

        if detection_method not in leakage_detection_methods.keys():
            raise ValueError(f"Invalid Detection Method {detection_method}")
        else:
            n_training_folds = self.cv_iterations - 1
            n_test_folds = 1
            model_results = self.read_results_file(detection_method)
            model_pvalues = {}
            for model_name, metric_vals in model_results.items():
                if 'MI' in detection_method:
                    base_mi = self.random_state.rand(len(metric_vals)) * 1e-2
                    p_value = paired_ttest(base_mi, metric_vals, n_training_folds, n_test_folds, correction=True)
                elif detection_method == PAIRED_TTEST:
                    accuracies = self.results[MAJORITY_VOTING]
                    p_value = paired_ttest(accuracies, metric_vals, n_training_folds, n_test_folds, correction=True)
                elif 'fishers' in detection_method:
                    p_values = np.array([fisher_exact(cm)[1] for cm in metric_vals])
                    if detection_method == FISHER_EXACT_TEST_MEAN:
                        p_value = np.mean(p_values)
                    elif detection_method == FISHER_EXACT_TEST_MEDIAN:
                        p_value = np.median(p_values)
                model_pvalues[model_name] = p_value
            p_vals, pvals_corrected, rejected = holm_bonferroni(list(model_pvalues.values()))
        return np.any(rejected), np.sum(rejected)
