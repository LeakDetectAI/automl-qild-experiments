import hashlib
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
from pycilt.detectors.utils import leakage_detection_methods, mi_estimation_metrics, calibrators, calibrator_params
from pycilt.metrics import probability_calibration
from pycilt.statistical_tests import paired_ttest
from pycilt.utils import log_exception_error, create_directory_safely


class InformationLeakageDetector(metaclass=ABCMeta):
    def __init__(self, padding_name, learner_params, fit_params, hash_value, cv_iterations, n_hypothesis,
                 base_directory, random_state, **kwargs):
        self.logger = logging.getLogger(InformationLeakageDetector.__name__)
        self.padding_name, self.padding_code = self.format_name(padding_name)
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
        self.results_file = os.path.join(self.base_directory, RESULT_FOLDER, f"{hash_value}_intermediate.h5")
        create_directory_safely(self.results_file, True)
        self.__initialize_objects__()

    def format_name(self, padding_name):
        padding_name = '_'.join(padding_name.split(' ')).lower()
        padding_name = padding_name.replace(" ", "")
        hash_object = hashlib.sha1()
        hash_object.update(padding_name.encode())
        hex_dig = str(hash_object.hexdigest())[:16]
        # self.logger.info(   "Job_id {} Hash_string {}".format(job.get("job_id", None), str(hex_dig)))
        self.logger.info(f"For padding name {padding_name} the hex value is {hex_dig}")
        return padding_name, hex_dig

    @property
    def _is_fitted_(self) -> bool:
        conditions = [os.path.exists(self.results_file)]
        file = None
        if os.path.exists(self.results_file):
            file = h5py.File(self.results_file, 'r')
            conditions.append(self.padding_code in file)
            if self.padding_code in file:
                self.logger.info(f"Simulations done for padding label {self.padding_code}")
                for model_name, metric_results in self.results.items():
                    padding_name_group = file[self.padding_code]
                    conditions.append(model_name in padding_name_group)
                    if model_name in padding_name_group:
                        model_group = padding_name_group.get(model_name)
                        self.logger.info(f"Predictions done for model {model_name}")
                        for metric_name, results in metric_results.items():
                            conditions.append(metric_name in model_group)
                            self.logger.info(f"Results exists for metric {metric_name}")
        if file is not None:
            file.close()
        if os.path.exists(self.results_file) and not np.all(conditions):
            if os.path.exists(self.results_file):
                file = h5py.File(self.results_file, 'r')
                if self.padding_code in file:
                    del file[self.padding_code]
                    self.logger.info(f"Results for padding {self.padding_name} removed since it is "
                                     f"incomplete {not np.all(conditions)}")
        if file is not None:
            file.close()
        self.close_file()
        return np.all(conditions)

    def __initialize_objects__(self):
        for i in range(self.n_hypothesis):
            self.results[f'model_{i}'] = {}
            for metric_name, evaluation_metric in mi_estimation_metrics.items():
                self.results[f'model_{i}'][metric_name] = []
        self.results[MAJORITY_VOTING] = {}
        self.results[MAJORITY_VOTING][ACCURACY] = []

    def get_training_dataset(self, X, y):
        lengths = []
        for i, (train_index, test_index) in enumerate(self.cv_iterator.split(X, y)):
            lengths.append(len(train_index))
        test_size = X.shape[0] - np.min(lengths)
        self.logger.info(f"Test size {test_size} Train sizes {lengths}")
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

    def close_file(self):
        is_open = False
        try:
            file = h5py.File(self.results_file, 'r')
            is_open = file.id.valid
            if is_open:
                self.logger.info("The file is open.")
                file.close()
            else:
                self.logger.info("The file is not open.")
        except Exception as error:
            log_exception_error(self.logger, error)
            self.logger.error("Cannot open the file since it does not exist")


    def evaluate_scores(self, X_test, X_train, y_test, y_train, y_pred, p_pred, model, i):
        for metric_name, evaluation_metric in mi_estimation_metrics.items():
            if LOG_LOSS_MI_ESTIMATION in metric_name or PC_SOFTMAX_MI_ESTIMATION in metric_name:
                calibrator_technique = None
                for key in calibrators.keys():
                    if key in metric_name:
                        calibrator_technique = key
                if calibrator_technique is not None:
                    calibrator = calibrators[calibrator_technique]
                    c_params = calibrator_params[calibrator_technique]
                    calibrator = calibrator(**c_params)
                    try:
                        p_pred_cal = probability_calibration(X_train, y_train, X_test, model, calibrator,
                                                             self.logger)
                        metric_loss = evaluation_metric(y_test, p_pred_cal)
                    except Exception as error:
                        log_exception_error(self.logger, error)
                        self.logger.error("Error while calibrating the probabilities")
                        metric_loss = evaluation_metric(y_test, p_pred)
                else:
                    metric_loss = evaluation_metric(y_test, p_pred)
            else:
                metric_loss = evaluation_metric(y_test, y_pred)
            if metric_name == CONFUSION_MATRIX:
                # metric_loss = np.array(metric_loss)
                (tn, fp, fn, tp) = metric_loss.ravel()
                cm_string = f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}"
                metric_loss = [tn, fp, fn, tp]
                self.logger.info(f"Metric {metric_name}: Value: {cm_string}")
            else:
                self.logger.info(f"Metric {metric_name}: Value: {metric_loss}")
            model_name = list(self.results.keys())[i]
            self.results[model_name][metric_name].append(metric_loss)

    def store_results(self):
        self.logger.info(f"Result file {self.results_file}")
        if os.path.exists(self.results_file):
            file = h5py.File(self.results_file, 'r+')
        else:
            file = h5py.File(self.results_file, 'w')
        try:
            self.logger.info(f"{self.padding_code} in {file}: {self.padding_code in file}")
            if self.padding_code not in file:
                padding_name_group = file.create_group(self.padding_code)
            else:
                padding_name_group = file.get(self.padding_code)
            for model_name, metric_results in self.results.items():
                self.logger.info(f"{model_name} in {padding_name_group}: {model_name in padding_name_group}")
                model_group = padding_name_group.create_group(model_name)
                self.logger.info(f"Creating model group {model_name} results {model_group}")
                for metric_name, results in metric_results.items():
                    self.logger.info(f"Storing results {metric_name} results {np.array(results)}")
                    model_group.create_dataset(metric_name, data=np.array(results))
        except Exception as error:
            log_exception_error(self.logger, error)
            self.logger.error("Problem creating the dataset ")
        finally:
            file.close()
        self.close_file()

    def allkeys(self, obj):
        "Recursively find all keys in an h5py.Group."
        keys = (obj.name,)
        if isinstance(obj, h5py.Group):
            for key, value in obj.items():
                if isinstance(value, h5py.Group):
                    keys = keys + self.allkeys(value)
                else:
                    keys = keys + (value.name,)
        return keys

    def read_results_file(self, detection_method):
        metric_name = leakage_detection_methods[detection_method]
        model_results = {}
        if os.path.exists(self.results_file):
            file = h5py.File(self.results_file, 'r')
            # self.logger.error(self.allkeys(file))
            padding_name_group = file[self.padding_code]
            # self.logger.error(self.allkeys(padding_name_group))
            for model_name in self.results.keys():
                if model_name == MAJORITY_VOTING:
                    continue
                model_group = padding_name_group[model_name]
                # self.logger.error(self.allkeys(model_group))
                try:
                    model_results[model_name] = np.array(model_group[metric_name])
                except KeyError as e:
                    log_exception_error(self.logger, e)
                    self.logger.error("Error while calibrating the probabilities")
                    raise ValueError(f"Provided Metric Name {metric_name} is not applicable "
                                     f"for current base detector {self.base_detector} "
                                     f"so cannot apply the provided detection method {detection_method}")
            file.close()
            self.close_file()
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
            model_p_values = {}
            for model_name, metric_vals in model_results.items():
                if 'MI' in detection_method:
                    base_mi = self.random_state.rand(len(metric_vals)) * 1e-2
                    p_value = paired_ttest(base_mi, metric_vals, n_training_folds, n_test_folds, correction=True)
                elif detection_method == PAIRED_TTEST:
                    accuracies = np.array(self.results[MAJORITY_VOTING][ACCURACY])
                    p_value = paired_ttest(accuracies, metric_vals, n_training_folds, n_test_folds, correction=True)
                elif 'fishers' in detection_method:
                    metric_vals = [np.array([[tn, fp], [fn, tp]]) for [tn, fp, fn, tp] in metric_vals]
                    p_values = np.array([fisher_exact(cm)[1] for cm in metric_vals])
                    if detection_method == FISHER_EXACT_TEST_MEAN:
                        p_value = np.mean(p_values)
                    elif detection_method == FISHER_EXACT_TEST_MEDIAN:
                        p_value = np.median(p_values)
                model_p_values[model_name] = p_value
            p_vals, pvals_corrected, rejected = holm_bonferroni(list(model_p_values.values()))
        return np.any(rejected), np.sum(rejected)
