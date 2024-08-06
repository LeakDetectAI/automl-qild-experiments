import inspect
import logging
import os
import shutil

from experiments.dbconnection import DBConnector
from experiments.utils import setup_logging
from autoqild import *

DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

if __name__ == "__main__":
    config_file_path = os.path.join(DIR_PATH, EXPERIMENTS, 'config', 'autosca.json')
    log_path = os.path.join(DIR_PATH, EXPERIMENTS, 'delete_files.log')
    setup_logging(log_path=log_path)
    logger = logging.getLogger('Experiment')
    logger.info(f"DB config filePath {config_file_path}")
    for schema in [AUTO_ML]:
        dbConnector = DBConnector(config_file_path=config_file_path, is_gpu=False, schema=schema,
                                  create_hash_list=True)
        dbConnector.init_connection()
        query = (f"SELECT hash_value, learning_problem, dataset_params, learner FROM {schema}.avail_jobs "
                 f"where dataset='synthetic_imbalanced' and learner = 'auto_gluon' order by job_id")
        dbConnector.cursor_db.execute(query)
        jobs_all = dbConnector.cursor_db.fetchall()
        for row in jobs_all:
            hash_value, learning_problem, dataset_params, learner = row
            BASE_DIR = os.path.join(DIR_PATH, EXPERIMENTS, learning_problem)
            log_path = os.path.join(BASE_DIR, LOGS_FOLDER, f"{hash_value}.log")
            optimizers_file_path = os.path.join(BASE_DIR, OPTIMIZER_FOLDER, f"{hash_value}.pkl")
            logger.info(f"Deleting files for {learner} params {dataset_params} hash_value {hash_value}")
            # Check if the folder exists and delete it
            if os.path.exists(log_path):
                os.remove(log_path)
                logger.info(f"Deleting log file {log_path}")

            # Check if the optimizer file exists and delete it
            if os.path.exists(optimizers_file_path):
                os.remove(optimizers_file_path)
                logger.info(f"Deleting optimizer {optimizers_file_path}")

            if learner == AUTO_GLUON:
                folder = os.path.join(BASE_DIR, OPTIMIZER_FOLDER, f"{hash_value}gluon")
                logger.info(f"Deleting optimizer {folder}")
                if os.path.exists(folder):
                    shutil.rmtree(folder)
