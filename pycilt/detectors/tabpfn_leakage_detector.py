import logging
import os.path

import dill
from sklearn.model_selection import StratifiedKFold

from .ild_base_class import InformationLeakageDetector
from .utils import mi_estimation_metrics, calibrators, calibrator_params
from .. import AutoTabPFNClassifier
from ..bayes_search_utils import get_scores
from ..contants import LOG_LOSS_MI_ESTIMATION
from ..metrics import probability_calibration
from ..multi_layer_perceptron import MultiLayerPerceptron
from ..utils import log_exception_error


class TabPFNLeakageDetector(InformationLeakageDetector):
    def __int__(self, learner_params, fit_params, result_folder, cv_iterations, random_state):
        super().__int__(cv_iterations=cv_iterations, random_state=random_state)
        self.learner_params = learner_params
        self.fit_params = fit_params
        self.result_folder = result_folder
        self.base_detector = AutoTabPFNClassifier
        self.logger = logging.getLogger(TabPFNLeakageDetector.__name__)

    def fit(self, X, y):
        skf = StratifiedKFold(n_splits=self.cv_iterations, random_state=0)
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            estimator = MultiLayerPerceptron(**self.learner_params)
            estimator.fit(X_train, y_train)
            self.estimators.append(estimator)
            # estimator.fit_pipeline(X_train, y_train, config=config, X_test=X_test, y_test=y_test)
            file_path = os.path.join(self.result_folder, f'learner_{i}')
            dill.dump(estimator, open(file_path, "wb"))
            p_pred, y_pred = get_scores(X_test, estimator)
            for metric_name in mi_estimation_metrics.keys():
                self.results[metric_name] = []
            for metric_name, evaluation_metric in mi_estimation_metrics.items():
                if LOG_LOSS_MI_ESTIMATION in metric_name:
                    calibrator_technique = None
                    for key in calibrators.keys():
                        if key in metric_name:
                            calibrator_technique = key
                    if calibrator_technique is not None:
                        calibrator = calibrators[calibrator_technique]
                        c_params = calibrator_params[calibrator_technique]
                        calibrator = calibrator(**c_params)
                        try:
                            p_pred_cal = probability_calibration(X_train, y_train, X_test, estimator, calibrator,
                                                                 self.logger)
                            metric_loss = evaluation_metric(y_test, p_pred_cal)
                        except Exception as error:
                            log_exception_error(self.logger, error)
                            self.logger.error("Error while calibrating the probabilities")
                            metric_loss = evaluation_metric(y_test, p_pred)
                        else:
                            metric_loss = evaluation_metric(y_test, p_pred)
                else:
                    metric_loss = evaluation_metric(y_test, p_pred)
                self.results[metric_name].append(metric_loss)
