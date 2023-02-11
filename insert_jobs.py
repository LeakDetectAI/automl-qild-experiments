import inspect
import logging
import os

from experiments.contants import *
from experiments.dbconnection import DBConnector
from experiments.util import setup_logging

DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
EXPERIMENTS = 'experiments'

if __name__ == "__main__":
    config_file_path = os.path.join(DIR_PATH, EXPERIMENTS, 'config', 'autosca.json')
    log_path = os.path.join(DIR_PATH, EXPERIMENTS, 'jobs_insertion.log')
    setup_logging(log_path=log_path)
    logger = logging.getLogger('Experiment')
    logger.info(f"DB config filePath {config_file_path}")
    for schema in [MUTUAL_INFORMATION_NEW]:
        logger.info(f"Inserting new jobs into {schema}")
        dbConnector = DBConnector(config_file_path=config_file_path, is_gpu=False, schema=schema, create_hash_list=True)
        if schema == CLASSIFICATION:
            max_job_id = 12
        if schema == MUTUAL_INFORMATION or schema == MUTUAL_INFORMATION_NEW:
            max_job_id = 4
        if schema == AUTO_ML:
            max_job_id = 1

        dbConnector.insert_new_jobs_different_configurations(max_job_id=max_job_id)
        dbConnector.insert_new_jobs_with_different_fold()
