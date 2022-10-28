from pycilt.utils import print_dictionary
import numpy as np

def update_params(bayes_search, logger, params):
    params.update(bayes_search.best_params_)
    params_str = print_dictionary(bayes_search.best_params_, sep='\t')
    best_score = bayes_search.best_score_
    logger.info(f"Best parameters are: {params_str} with accuracy of: {best_score}\n")
    return params


def log_callback(logger):
    def on_step(opt_result):
        """
        Callback meant to view scores after
        each iteration while performing Bayesian
        Optimization in Skopt"""
        points = opt_result.x_iters
        scores = -opt_result.func_vals
        logger.info('Next parameters: {}, accuracy {}'.format(points[-1], scores[-1]))

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
            p_pred = np.max(pred_prob, axis=1)
    else:
        p_pred = pred_prob.flatten()
    return p_pred, y_pred
