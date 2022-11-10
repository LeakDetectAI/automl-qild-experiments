import inspect
import logging
import os

from experiments.contants import MUTUAL_INFORMATION, CLASSIFICATION
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
    for schema in [CLASSIFICATION, MUTUAL_INFORMATION]:
        logger.info(f"Inserting new jobs into {schema}")
        dbConnector = DBConnector(config_file_path=config_file_path, is_gpu=False, schema=schema)
        dbConnector.insert_new_jobs_different_configurations()
        dbConnector.insert_new_jobs_with_different_fold()
