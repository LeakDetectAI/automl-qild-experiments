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

from pycilt.utils import setup_logging, setup_random_seed, print_dictionary
from util import get_duration_seconds
from dbconnection import DBConnector


DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
LOGS_FOLDER = 'logs'
OPTIMIZER_FOLDER = 'optimizers'
PREDICTIONS_FOLDER = 'predictions'
MODEL_FOLDER = 'models'
ERROR_OUTPUT_STRING = 'Out of sample error %s : %0.4f'


if __name__ == "__main__":
    start = datetime.now()

    ######################## DOCOPT ARGUMENTS: #################################
    arguments = docopt(__doc__)
    cluster_id = int(arguments["--cindex"])
    is_gpu = bool(int(arguments["--isgpu"]))
    schema = arguments["--schema"]
    ###################### POSTGRESQL PARAMETERS ###############################
    config_file_path = os.path.join(DIR_PATH, 'config', 'autosca.json')
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
            log_path = os.path.join(DIR_PATH, LOGS_FOLDER, "{}.log".format(hash_value))
            setup_logging(log_path=log_path)
            setup_random_seed(seed=seed)
            logger = logging.getLogger('Experiment')
            logger.info("DB config filePath {}".format(config_file_path))
            logger.info("Arguments {}".format(arguments))
            logger.info("Job Description {}".format(print_dictionary(dbConnector.job_description)))
            duration = get_duration_seconds(duration)
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