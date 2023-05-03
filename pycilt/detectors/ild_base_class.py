import logging
from abc import ABCMeta

import numpy as np
from scipy.stats import fisher_exact

from pycilt.contants import MAJORITY_VOTING, PAIRED_TTEST, FISHER_EXACT_TEST_MEAN, FISHER_EXACT_TEST_MEDIAN
from pycilt.detectors.utils import leakage_detection_methods
from pycilt.statistical_tests import paired_ttest
from pycilt.utils import log_exception_error


class InformationLeakageDetector(metaclass=ABCMeta):
    def __int__(self, cv_iterations, random_state):
        self.cv_iterations = cv_iterations
        self.random_state = random_state
        self.estimators = []
        self.results = {MAJORITY_VOTING: []}
        self.base_detector = None
        self.logger = logging.getLogger(InformationLeakageDetector.__name__)

    def fit(self, X, y):
        raise NotImplemented

    def detect(self, detection_method):
        metric_name = leakage_detection_methods[detection_method]
        try:
            metric_vals = self.results[metric_name]
            n_training_folds = len(metric_vals) - 1
            n_test_folds = 1
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
            if p_value < 0.05:
                return True
            else:
                return False
        except KeyError as e:
            log_exception_error(self.logger, e)
            self.logger.error("Error while calibrating the probabilities")
            metric_vals = None
            raise ValueError(f"Provided Detection Method {detection_method} is not applicable "
                             f"for current base detector {self.base_detector}")
