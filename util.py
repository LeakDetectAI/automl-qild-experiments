import re
from datetime import datetime, timedelta

from pycilt.synthetic_data_generator import SyntheticDatasetGenerator
from contants import SYNTHETIC_DATASET

__all__ = ["get_dataset_reader", "duration_till_now", "time_from_now", "get_dataset_reader"]

datasets = {
    SYNTHETIC_DATASET: SyntheticDatasetGenerator,

}

def get_duration_seconds(duration):
    time = int(re.findall(r"\d+", duration)[0])
    d = duration.split(str(time))[1].upper()
    options = {"D": 24 * 60 * 60, "H": 60 * 60, "M": 60}
    return options[d] * time

def duration_till_now(start):
    return (datetime.now() - start).total_seconds()


def time_from_now(target_time_sec):
    base_datetime = datetime.now()
    delta = timedelta(seconds=target_time_sec)
    target_date = base_datetime + delta
    return target_date.strftime("%Y-%m-%d %H:%M:%S")

def get_dataset_reader(dataset_name, dataset_params):
    dataset_func = datasets[dataset_name]
    dataset_func = dataset_func(**dataset_params)
    return dataset_func
