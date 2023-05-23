import logging

from sklearn.model_selection import StratifiedShuffleSplit

from .ild_base_class import InformationLeakageDetector
from .utils import mi_estimation_metrics, calibrators, calibrator_params
from ..bayes_search import BayesSearchCV
from ..bayes_search_utils import get_scores, log_callback, update_params_at_k
from ..contants import LOG_LOSS_MI_ESTIMATION, PC_SOFTMAX_MI_ESTIMATION
from ..metrics import probability_calibration
from ..utils import log_exception_error


class SklearnClassifierLeakageDetector(InformationLeakageDetector):
    def __int__(self, padding_name, learner_params, fit_params, hash_value, cv_iterations, n_hypothesis, base_directory,
                search_space, hp_iters, n_inner_folds, validation_loss, random_state=None, **kwargs):
        super().__int__(padding_name=padding_name, learner_params=learner_params, fit_params=fit_params,
                        hash_value=hash_value, cv_iterations=cv_iterations, n_hypothesis=n_hypothesis,
                        base_directory=base_directory, random_state=random_state, **kwargs)
        self.search_space = search_space
        self.hp_iters = hp_iters
        self.n_inner_folds = n_inner_folds
        self.validation_loss = validation_loss
        self.inner_cv_iterator = StratifiedShuffleSplit(n_splits=self.n_inner_folds, test_size=0.10,
                                                        random_state=self.random_state)
        self.logger = logging.getLogger(SklearnClassifierLeakageDetector.__name__)
        self.n_jobs = 10

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
        callback = log_callback(self.logger, search_keys)
        try:
            bayes_search.fit(X_train, y_train, groups=None, callback=callback, **self.fit_params)
        except Exception as error:
            log_exception_error(self.logger, error)
            self.logger.error("Cannot fit the Bayes SearchCV ")
        train_size = X_train.shape[0]
        for i in range(self.n_hypothesis):
            loss, learner_params = update_params_at_k(bayes_search, search_keys, self.learner_params, self.logger, k=0)
            learner = self.base_detector(**learner_params)
            self.logger.info(f"Model {i} with loss {loss} and parameters {learner_params}")
            self.estimators.append(learner)
        return train_size

    def fit(self, X, y):
        if self._is_fitted_:
            self.logger.info(f"Model already fitted for the padding {self.padding_name}")
        else:
            train_size = self.perform_hyperparameter_optimization(X, y)
            for k, (train_index, test_index) in enumerate(self.cv_iterator.split(X, y)):
                self.logger.info(f"************************************ Split {k} ************************************")
                train_index = train_index[:train_size]
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                self.calculate_majority_voting_accuracy(X_train, y_train, X_test, y_test)
                for i, model in enumerate(self.estimators):
                    self.logger.info(
                        f"************************************ Model {i} ************************************")
                    model.fit(X=X_train, y=y_train)
                    p_pred, y_pred = get_scores(X_test, model)
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
                        self.logger.info(f"Metric {metric_name}: Value {metric_loss}")
                        model_name = list(self.results.keys())[i]
                        self.results[model_name][metric_name].append(metric_loss)
            self.store_results()
