import inspect
import logging
import os

from experiments.dbconnection import DBConnector
from experiments.utils import setup_logging
from pycilt.contants import *

DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
EXPERIMENTS = 'experiments'

if __name__ == "__main__":
    config_file_path = os.path.join(DIR_PATH, EXPERIMENTS, 'config', 'autosca.json')
    log_path = os.path.join(DIR_PATH, EXPERIMENTS, 'jobs_insertion.log')
    setup_logging(log_path=log_path)
    logger = logging.getLogger('Experiment')
    logger.info(f"DB config filePath {config_file_path}")
    for schema in [AUTO_ML]:
        logger.info(f"Inserting new jobs into {schema}")
        dbConnector = DBConnector(config_file_path=config_file_path, is_gpu=False, schema=schema, create_hash_list=True)
        if schema == CLASSIFICATION:
            max_job_id = 12
        if schema == MUTUAL_INFORMATION:
            max_job_id = 2
        if schema == MUTUAL_INFORMATION_NEW:
            max_job_id = 5
        if schema == AUTO_ML:
            max_job_id = 4

        dbConnector.insert_new_jobs_different_configurations(max_job_id=max_job_id, dataset=SYNTHETIC_DATASET)
        dbConnector.insert_new_jobs_with_different_fold(dataset=SYNTHETIC_DATASET)
        dbConnector.insert_new_jobs_different_configurations(max_job_id=max_job_id, dataset=SYNTHETIC_DISTANCE_DATASET)
        dbConnector.insert_new_jobs_with_different_fold(dataset=SYNTHETIC_DISTANCE_DATASET)
