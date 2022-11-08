import numpy as np

from pycilt.utils import print_dictionary


def get_parameters(optimizers, search_keys):
    yis = []
    xis = []
    for opt in optimizers:
        yis.extend(opt.yi)
        xis.extend(opt.Xi)
    best_i = np.argmin(yis)
    best_params = xis[best_i]
    best_loss = yis[best_i]
    best_params = dict(zip(search_keys, best_params))
    return best_loss, best_params


def update_params(bayes_search, search_keys, params, logger):
    best_loss, best_params = get_parameters(bayes_search.optimizers_, search_keys)
    params.update(best_params)
    params_str = print_dictionary(best_params, sep='\t')
    logger.info(f"Best parameters are: {params_str} with Accuracy/MI of: {-best_loss}\n")
    return params


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
        if pred_prob.shape[-1] == 2:
            p_pred = pred_prob[:, 1]
        else:
            p_pred = pred_prob
    else:
        p_pred = pred_prob.flatten()
    return p_pred, y_pred
