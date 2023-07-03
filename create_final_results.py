import inspect
import logging
import os

import numpy as np

from experiments.dbconnection import DBConnector
from experiments.utils import get_dataset_reader, setup_logging, insert_results_in_table, create_results, \
    lp_metric_dict
from pycilt.utils import print_dictionary

DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
EXPERIMENTS = 'experiments'

if __name__ == "__main__":
    log_path = os.path.join(DIR_PATH, EXPERIMENTS, 'create_final_results.log')
    setup_logging(log_path=log_path)
    logger = logging.getLogger('Experiment')
    config_file_path = os.path.join(DIR_PATH, EXPERIMENTS, 'config', 'autosca.json')
    db_connector = DBConnector(config_file_path=config_file_path, is_gpu=False)
    db_connector.init_connection()
    schema = 'leakage_detection_new'
    result_table = f"results.{schema}"
    avail_jobs = f"{schema}.avail_jobs"
    final_result_table = f"results.{schema}_finale"
    select_job = f"""SELECT * FROM {result_table} JOIN {avail_jobs} ON {result_table}.job_id = {avail_jobs}.job_id 
                     order by {result_table}.job_id;"""
    db_connector.cursor_db.execute(select_job)
    final_results = db_connector.cursor_db.fetchall()
    drop_table = f"DROP table {final_result_table}"
    db_connector.cursor_db.execute(drop_table)
    drop_table = f"DROP table results.leakage_detection_new_final"
    db_connector.cursor_db.execute(drop_table)
    create_table = f"""CREATE TABLE IF NOT EXISTS {final_result_table}
                    (
                        job_id                        integer not null,
                        n_hypothesis_threshold        integer not null,
                        dataset_id                    integer not null,
                        cluster_id                    integer not null,
                        base_detector                 text    not null,
                        detection_method              text    not null,
                        fold_id                       integer not null,
                        imbalance                     double precision,
                        delay                         double precision,
                        f1score                       double precision,
                        accuracy                      double precision,
                        mathewscorrelationcoefficient double precision,
                        balancedaccuracy              double precision,
                        falsepositiverate             double precision,
                        falsenegativerate             double precision,
                        hypothesis                    json
                    );"""
    db_connector.cursor_db.execute(create_table)
    admin_allocation = f"""alter table {final_result_table} owner to autoscaadmin;"""
    db_connector.cursor_db.execute(admin_allocation)
    column_names_query = f"SELECT * FROM {final_result_table} LIMIT 0;"
    db_connector.cursor_db.execute(column_names_query)
    column_names = [desc[0] for desc in db_connector.cursor_db.description]
    db_connector.connection.commit()
    for result in final_results:
        dataset_name = result["dataset"]
        dataset_params = result["dataset_params"]
        learning_problem = result["learning_problem"]
        dataset_reader = get_dataset_reader(dataset_name, dataset_params)
        results_new = create_results(result)
        hypothesis = dict(result['hypothesis'])
        logger.info(f"Creating results from {print_dictionary(result)}")
        for threshold in np.arange(1, 11):
            y_true, y_pred = [], []
            for label in dataset_reader.label_mapping.keys():
                if label == dataset_reader.correct_class:
                    continue
                ground_truth = int(label in dataset_reader.vulnerable_classes)
                y_true.append(ground_truth)
                rejected_hypothesis = hypothesis[label]
                y_pred.append(int(rejected_hypothesis > threshold))
            results_new['n_hypothesis_threshold'] = threshold
            for metric_name, evaluation_metric in lp_metric_dict[learning_problem].items():
                metric_loss = evaluation_metric(y_true, y_pred)
                if np.isnan(metric_loss) or np.isinf(metric_loss):
                    results_new[metric_name] = "\'Infinity\'"
                else:
                    if np.around(metric_loss, 4) == 0.0:
                        results_new[metric_name] = f"{metric_loss}"
                    else:
                        results_new[metric_name] = f"{np.around(metric_loss, 4)}"
            logger.info(f"Results for threshold {threshold} is: {print_dictionary(results_new)}")
            insert_results_in_table(db_connector, results_new, final_result_table, logger)
        db_connector.connection.commit()
    db_connector.close_connection()
