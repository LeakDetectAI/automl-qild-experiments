import numpy as np
import sklearn
from autogluon.core.models import AbstractModel
from packaging import version
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC

from pycilt.utils import print_dictionary, sigmoid


def get_parameters_at_k(optimizers, search_keys, k):
    yis = []
    xis = []
    for opt in optimizers:
        yis.extend(opt.yi)
        xis.extend(opt.Xi)
    yis = np.array(yis)
    xis = np.array(xis)
    index_k = np.argsort(yis)[k]
    best_params = xis[index_k]
    best_loss = yis[index_k]
    best_params = dict(zip(search_keys, best_params))
    return best_loss, best_params


def update_params_at_k(bayes_search, search_keys, params, logger, k=0):
    loss, best_params = get_parameters_at_k(optimizers=bayes_search.optimizers_, search_keys=search_keys, k=k)
    if version.parse(sklearn.__version__) < version.parse("0.25.0"):
        if 'criterion' in best_params.keys():
            if best_params['criterion'] == 'squared_error':
                best_params['criterion'] = 'mse'
    params.update(best_params)
    params_str = print_dictionary(best_params, sep='\t')
    logger.info(f"Parameters at position k:{k} are {params_str} with objective of: {-loss}\n")
    return loss, params


def log_callback(logger, parameters):
    def on_step(opt_result):
        """
        Callback meant to view scores after
        each iteration while performing Bayesian
        Optimization in Skopt"""
        points = opt_result.x_iters
        scores = -opt_result.func_vals
        params = dict(zip(parameters, points[-1]))
        params_str = print_dictionary(params, sep=' : ')
        logger.info(f'For Parameters: {params_str}, Objective: {scores[-1]}')
    return on_step


def get_scores(X, estimator):
    y_pred = estimator.predict(X)
    try:
        pred_prob = estimator.predict_proba(X)
    except:
        pred_prob = estimator.decision_function(X)
    # logger.info("Predict Probability shape {}, {}".format(pred_prob.shape, y_test.shape))
    if len(pred_prob.shape) == 2 and pred_prob.shape[-1] > 1:
        p_pred = pred_prob
    else:
        p_pred = pred_prob.flatten()
    if isinstance(estimator, AbstractModel):
        if len(p_pred.shape) == 1:
            p_pred = np.hstack(((1 - p_pred)[:, None], p_pred[:, None]))
    if isinstance(estimator, SGDClassifier) or isinstance(estimator, LinearSVC) or isinstance(estimator,
                                                                                              RidgeClassifier):
        p_pred = sigmoid(p_pred)
        if len(p_pred.shape) == 1:
            p_pred = np.hstack(((1 - p_pred)[:, None], p_pred[:, None]))
    y_pred = np.array(y_pred)
    p_pred = np.array(p_pred)
    #logger = logging.getLogger("Score")
    #logger.info(f"Scores Shape {p_pred.shape}, Classes {np.unique(y_pred)}")
    return p_pred, y_pred
