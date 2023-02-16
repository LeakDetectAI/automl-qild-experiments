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

import h5py
import numpy as np
from autosklearn.estimators import AutoSklearnClassifier
from docopt import docopt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from experiments.bayes_search_utils import update_params, log_callback, get_scores
from experiments.contants import EMI, F_SCORE
from experiments.dbconnection import DBConnector
from experiments.util import get_duration_seconds, get_dataset_reader, create_search_space, setup_logging, \
    setup_random_seed, create_directory_safely, learners, lp_metric_dict, convert_learner_params, duration_till_now, \
    seconds_to_time
from pycilt.bayes_predictor import BayesPredictor
from pycilt.bayes_search import BayesSearchCV
from pycilt.mi_estimators import GMMMIEstimator, PCSoftmaxMIEstimator, MineMIEstimator
from pycilt.mi_estimators.mi_base_class import MIEstimatorBase
from pycilt.multi_layer_perceptron import MultiLayerPerceptron
from pycilt.utils import print_dictionary

DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
LOGS_FOLDER = 'logs'
RESULT_FOLDER = 'results'
EXPERIMENTS = 'experiments'
OPTIMIZER_FOLDER = 'optimizers'


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
    os.environ["HIP_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for i in range(10):
        dbConnector.job_description = None
        if 'CCS_REQID' in os.environ.keys():
            cluster_id = int(os.environ['CCS_REQID'])
        print("**************************************************************************************************")
        print(f"Getting the job for iteration {i}")
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
                learner_params_db = dbConnector.job_description["learner_params"]
                duration = dbConnector.job_description["duration"]
                hp_iters = int(dbConnector.job_description["hp_iters"])
                hp_ranges = dbConnector.job_description["hp_ranges"]
                learning_problem = dbConnector.job_description["learning_problem"]
                experiment_schema = dbConnector.job_description["experiment_schema"]
                experiment_table = dbConnector.job_description["experiment_table"]
                validation_loss = dbConnector.job_description["validation_loss"]
                hash_value = dbConnector.job_description["hash_value"]
                LEARNING_PROBLEM = learning_problem.lower()
                if validation_loss == 'None':
                    validation_loss = None
                random_state = np.random.RandomState(seed=seed + fold_id)
                log_path = os.path.join(DIR_PATH, EXPERIMENTS, LEARNING_PROBLEM, LOGS_FOLDER, f"{hash_value}.log")
                base_dir = os.path.join(DIR_PATH, EXPERIMENTS, LEARNING_PROBLEM)
                create_directory_safely(base_dir, False)
                create_directory_safely(log_path, True)

                setup_logging(log_path=log_path)
                setup_random_seed(random_state=random_state)
                logger = logging.getLogger('Experiment')
                logger.info(f"DB config filePath {config_file_path}")
                logger.info(f"Arguments {arguments}")
                logger.info(f"Job Description {print_dictionary(dbConnector.job_description)}")
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

                optimizers_file_path = os.path.join(DIR_PATH, EXPERIMENTS, LEARNING_PROBLEM, OPTIMIZER_FOLDER,
                                                    f"{hash_value}.pkl")
                create_directory_safely(optimizers_file_path, True)

                learner = learners[learner_name]
                learner_params = convert_learner_params(learner_params_db)
                learner_params['random_state'] = random_state
                logger.info(f"Time Taken till now: {seconds_to_time(duration_till_now(start))}  seconds")

                if learner == MultiLayerPerceptron or issubclass(learner, MIEstimatorBase):
                    n_jobs = 1
                    if learner == GMMMIEstimator:
                        n_jobs = 10
                else:
                    n_jobs = 10
                if learner == BayesPredictor or issubclass(learner, DummyClassifier) \
                        or learner == AutoSklearnClassifier:
                    if learner == BayesPredictor:
                        learner_params = {'dataset_obj': dataset_reader}
                        estimator = learner(**learner_params)
                        estimator.fit(X_train, y_train)
                        y_true, y_pred, p_pred = estimator.get_bayes_predictor_scores()
                    elif issubclass(learner, DummyClassifier):
                        estimator = learner(**learner_params)
                        estimator.fit(X_train, y_train)
                        p_pred, y_pred = get_scores(X, estimator)
                        y_true = np.copy(y)
                    else:
                        import shutil
                        learner_params['tmp_folder'] = os.path.join(DIR_PATH, EXPERIMENTS, LEARNING_PROBLEM,
                                                                    OPTIMIZER_FOLDER, hash_value)
                        if os.path.exists(learner_params['tmp_folder']):
                            logger.info(f"Removing the existing directory {learner_params['tmp_folder']} ")
                            shutil.rmtree(learner_params['tmp_folder'])
                        del learner_params['random_state']
                        estimator = learner(**learner_params)
                        estimator.fit(X_train, y_train)
                        config = estimator.get_configuration_space(X_train, y_train)
                        # estimator.fit_pipeline(X_train, y_train, config=config, X_test=X_test, y_test=y_test)
                        p_pred, y_pred = get_scores(X_test, estimator)
                        y_true = np.copy(y_test)
                        setup_logging(log_path=log_path)
                        logger.info(f"Best ensemble {estimator.leaderboard()}")
                        logger.info(f"Test Scores: {estimator.cv_results_['mean_test_score']}")
                        logger.info(f"Best Model {estimator.show_models()}")

                else:
                    inner_cv_iterator = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)
                    search_space = create_search_space(hp_ranges)
                    if learner == MultiLayerPerceptron or issubclass(learner, MIEstimatorBase):
                        learner_params = {**learner_params, **dict(input_dim=input_dim, n_classes=n_classes)}
                    if learner == GMMMIEstimator:
                        learner_params['n_models'] = 4
                    estimator = learner(**learner_params)
                    bayes_search_params = dict(estimator=estimator, search_spaces=search_space, n_iter=hp_iters,
                                               scoring=validation_loss, n_jobs=n_jobs, cv=inner_cv_iterator,
                                               error_score=0, random_state=random_state,
                                               optimizers_file_path=optimizers_file_path)
                    bayes_search = BayesSearchCV(**bayes_search_params)
                    search_keys = list(search_space.keys())
                    search_keys.sort()
                    logger.info(f"Search Keys {search_keys}")
                    callback = log_callback(logger, search_keys)
                    try:
                        bayes_search.fit(X_train, y_train, groups=None, callback=callback, **fit_params)
                    except Exception as e:
                        logger.info(f"Exception {str(e)}")
                    logger.info("Fitting the model with best parameters")
                    learner_params = update_params(bayes_search, search_keys, learner_params, logger)
                    logger.info(f"Setting the best parameters {print_dictionary(learner_params)}")
                    if learner == GMMMIEstimator:
                        del learner_params['n_models']
                    if learner == MineMIEstimator or learner == PCSoftmaxMIEstimator:
                        estimator = learner(**learner_params)
                        for _ in range(10):
                            estimator.fit(X_train, y_train)
                            logger.info(f"Final Loss {estimator.final_loss}")
                            if not (np.isnan(estimator.final_loss) or np.isinf(estimator.final_loss)):
                                break
                            else:
                                logger.info("Final Loss NAN train again")
                    else:
                        estimator = learner(**learner_params)
                        estimator.fit(X_train, y_train)
                    p_pred, y_pred = get_scores(X_test, estimator)
                    y_true = np.copy(y_test)
                if issubclass(learner, MIEstimatorBase):
                    estimated_mi = estimator.estimate_mi(X, y)
                else:
                    estimated_mi = 0
                result_file = os.path.join(DIR_PATH, EXPERIMENTS, LEARNING_PROBLEM, RESULT_FOLDER, f"{hash_value}.h5")
                logger.info(f"Result file {result_file}")
                create_directory_safely(result_file, True)
                f = h5py.File(result_file, 'w')
                f.create_dataset('scores', data=p_pred)
                f.create_dataset('predictions', data=y_pred)
                f.create_dataset('ground_truth', data=y_true)
                f.create_dataset('confusion_matrix', data=confusion_matrix(y_true, y_pred))
                f.close()

                results = {'job_id': str(job_id), 'cluster_id': str(cluster_id)}
                for name, evaluation_metric in lp_metric_dict[learning_problem].items():
                    if name == EMI:
                        metric_loss = estimated_mi
                    else:
                        if name == F_SCORE:
                            if n_classes > 2:
                                metric_loss = evaluation_metric(y_true, y_pred, average='macro')
                            else:
                                metric_loss = evaluation_metric(y_true, y_pred)
                        else:
                            metric_loss = evaluation_metric(y_true, y_pred)
                    if np.isnan(metric_loss) or np.isinf(metric_loss):
                        results[name] = "\'Infinity\'"
                    else:
                        if np.around(metric_loss, 4) == 0.0:
                            results[name] = f"{metric_loss}"
                        else:
                            results[name] = f"{np.around(metric_loss, 4)}"
                        # if CONFUSION_MATRIX == name:
                        #    tn, fp, fn, tp = metric_loss.ravel()
                        #   results['TN'] = "{0:.4f}".format(tn)
                        #   results['FP'] = "{0:.4f}".format(fp)
                        #   results['FN'] = "{0:.4f}".format(fn)
                        #   results['TP'] = "{0:.4f}".format(tp)
                    logger.info(f"Out of sample error {name} : {metric_loss}")
                    print(f"Out of sample error {name} : {metric_loss}")
                dbConnector.insert_results(experiment_schema=experiment_schema, experiment_table=experiment_table,
                                           results=results)
                dbConnector.mark_running_job_finished(job_id, start)
                logger.info(f"Job finished")
                print(f"Job finished")
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
