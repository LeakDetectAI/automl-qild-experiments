"""Thin experiment runner which takes all simulation parameters from a database.

Usage:
  experiment_cv.py --cindex=<id> --isgpu=<bool> --schema=<schema>
  experiment_cv.py (-h | --help)

Arguments:
  FILE                  An argument for passing in a file.

Options:
  -h --help                             Show this screen.
  --cindex=<cindex>                     Index given by the cluster to specify which job
                                        is to be executed [default: 0]
  --isgpu=<bool>                        Boolean to show if the gpu is to be used or not
  --schema=<schema>                     Schema containing the job information
"""
import inspect
import logging
import os
import sys
import traceback
from datetime import datetime

import numpy as np
from docopt import docopt
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from skopt import BayesSearchCV

from experiments.bayes_search_utils import update_params, log_callback, get_scores
from experiments.dbconnection import DBConnector
from experiments.util import get_duration_seconds, get_dataset_reader, create_search_space, setup_logging, \
    setup_random_seed, create_directory_safely, learners, lp_metric_dict, convert_learner_params
from pycilt.bayes_predictor import BayesPredictor
from pycilt.multi_layer_perceptron import MultiLayerPerceptron
from pycilt.utils import print_dictionary

DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
LOGS_FOLDER = 'logs'
RESULT_FOLDER = 'results'
EXPERIMENTS = 'experiments'

if __name__ == "__main__":
    start = datetime.now()

    ######################## DOCOPT ARGUMENTS: #################################
    arguments = docopt(__doc__)
    cluster_id = int(arguments["--cindex"])
    is_gpu = bool(int(arguments["--isgpu"]))
    schema = arguments["--schema"]
    ###################### POSTGRESQL PARAMETERS ###############################
    config_file_path = os.path.join(DIR_PATH, EXPERIMENTS, 'config', 'autosca.json')
    dbConnector = DBConnector(config_file_path=config_file_path, is_gpu=is_gpu, schema=schema)
    if 'CCS_REQID' in os.environ.keys():
        cluster_id = int(os.environ['CCS_REQID'])
    dbConnector.fetch_job_arguments(cluster_id=cluster_id)
    if dbConnector.job_description is not None:
        try:
            seed = int(dbConnector.job_description["seed"])
            job_id = int(dbConnector.job_description["job_id"])
            fold_id = int(dbConnector.job_description["fold_id"])
            dataset_name = dbConnector.job_description["dataset"]
            n_inner_folds = int(dbConnector.job_description["inner_folds"])
            dataset_params = dbConnector.job_description["dataset_params"]
            learner_name = dbConnector.job_description["learner"]
            fit_params = dbConnector.job_description["fit_params"]
            learner_params = dbConnector.job_description["learner_params"]
            duration = dbConnector.job_description["duration"]
            hp_iters = int(dbConnector.job_description["hp_iters"])
            hp_ranges = dbConnector.job_description["hp_ranges"]
            learning_problem = dbConnector.job_description["learning_problem"]
            experiment_schema = dbConnector.job_description["experiment_schema"]
            experiment_table = dbConnector.job_description["experiment_table"]
            validation_loss = dbConnector.job_description["validation_loss"]
            hash_value = dbConnector.job_description["hash_value"]
            random_state = np.random.RandomState(seed=seed + fold_id)
            log_path = os.path.join(DIR_PATH, EXPERIMENTS, LOGS_FOLDER, "{}.log".format(hash_value))
            setup_logging(log_path=log_path)
            setup_random_seed(seed=seed)
            logger = logging.getLogger('Experiment')
            logger.info("DB config filePath {}".format(config_file_path))
            logger.info("Arguments {}".format(arguments))
            logger.info("Job Description {}".format(print_dictionary(dbConnector.job_description)))
            duration = get_duration_seconds(duration)
            dataset_params['random_state'] = random_state
            dataset_params['fold_id'] = fold_id
            dataset_reader = get_dataset_reader(dataset_name, dataset_params)
            X, y = dataset_reader.generate_dataset()
            input_dim = X.shape[-1]
            n_classes = len(np.unique(y))
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=0)
            train_index, test_index = list(sss.split(X, y))[0]
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            inner_cv_iterator = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)
            search_spaces = create_search_space(hp_ranges)
            hash_file = os.path.join(DIR_PATH, RESULT_FOLDER, "{}.h5".format(hash_value))
            create_directory_safely(hash_file, True)
            learner = learners[learner_name]
            learner_params = convert_learner_params(learner_params)
            learner_params['random_state'] = random_state
            if learner == BayesPredictor or issubclass(learner, DummyClassifier):
                if learner == BayesPredictor:
                    learner_params = {'dataset_obj': dataset_reader}
                estimator = learner(**learner_params)
                estimator.fit(X_train, y_train)
                p_pred, y_pred = get_scores(X, estimator)
                y_true = y
            else:
                learner_params['random_state'] = random_state
                if learner == MultiLayerPerceptron:
                    learner_params = {**learner_params, **dict(input_dim=input_dim, n_classes=n_classes)}
                estimator = learner(**learner_params)
                bayes_search_params = dict(estimator=estimator, search_spaces=search_spaces, n_iter=hp_iters,
                                           scoring=validation_loss, n_jobs=10, cv=inner_cv_iterator, error_score=0,
                                           random_state=random_state)
                bayes_search = BayesSearchCV(**bayes_search_params)
                bayes_search.fit(X_train, y_train, groups=None, callback=log_callback(logger), **fit_params)
                learner_params = update_params(bayes_search, logger, learner_params)
                estimator = learner(**learner_params)
                estimator.fit(X_train, y_train)
                p_pred, y_pred = get_scores(X_test, estimator)
                y_true = y_test

            results = {'job_id': str(job_id), 'cluster_id': str(cluster_id)}
            for name, evaluation_metric in lp_metric_dict[learning_problem].items():
                predictions = y_pred
                if 'AUC' in name:
                    if n_classes>2:
                        metric_loss = evaluation_metric(y_true, p_pred, multi_class='ovr')
                    else:
                        metric_loss = evaluation_metric(y_true, p_pred)
                else:
                    metric_loss = evaluation_metric(y_true, y_pred)
                if 'ConfusionMatrix' == name:
                    tn, fp, fn, tp = metric_loss.ravel()
                    results['TN'] = "{0:.4f}".format(tn)
                    results['FP'] = "{0:.4f}".format(fp)
                    results['FN'] = "{0:.4f}".format(fn)
                    results['TP'] = "{0:.4f}".format(tp)
                else:
                    if np.isnan(metric_loss):
                        results[name] = "\'Infinity\'"
                    else:
                        results[name] = f"{np.around(metric_loss, 4)}"
                logger.info(f"Out of sample error {name} : {metric_loss}")

            dbConnector.insert_results(experiment_schema=experiment_schema, experiment_table=experiment_table,
                                       results=results)
            dbConnector.mark_running_job_finished(job_id)
        except Exception as e:
            if hasattr(e, 'message'):
                message = e.message
            else:
                message = e
            logger.error(traceback.format_exc())
            message = "exception{}".format(str(message))
            dbConnector.append_error_string_in_running_job(job_id=job_id, error_message=message)
        except:
            logger.error(traceback.format_exc())
            message = "exception{}".format(sys.exc_info()[0].__name__)
            dbConnector.append_error_string_in_running_job(job_id=job_id, error_message=message)
        finally:
            if "224" in str(cluster_id):
                f = open("{}/.hash_value".format(os.environ['HOME']), "w+")
                f.write(hash_value + "\n")
                f.close()