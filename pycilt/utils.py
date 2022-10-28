import inspect
import logging
import multiprocessing
import numpy as np
import os
import random
import sys
import tensorflow as tf
import warnings
from keras import backend as K
from sklearn.preprocessing import RobustScaler
from tensorflow.core.protobuf.config_pb2 import ConfigProto
from tensorflow.python.client.session import Session

warnings.filterwarnings('ignore')

__all__ = ['create_dir_recursively', 'setup_logging', 'progress_bar', 'print_dictionary',
           'standardize_features', 'standardize_features']

def progress_bar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s/%s ...%s\r' % (bar, count, total, status))
    sys.stdout.flush()


def print_dictionary(dictionary, sep='\n'):
    output = "\n"
    for key, value in dictionary.items():
        output = output + str(key) + " => " + str(value) + sep
    return output

def standardize_features(x_train, x_test):
    standardize = Standardize()
    x_train = standardize.fit_transform(x_train)
    x_test = standardize.transform(x_test)
    return x_train, x_test


class Standardize(object):
    def __init__(self, scalar=RobustScaler):
        self.scalar = scalar
        self.n_features = None
        self.scalars = dict()

    def fit(self, X):
        if isinstance(X, dict):
            self.n_features = list(X.keys())
            for k, x in X.items():
                scalar = self.scalar()
                self.scalars[k] = scalar.fit(x)
        if isinstance(X, (np.ndarray, np.generic)):
            self.scalar = self.scalar()
            self.scalar.fit(X)
            self.n_features = X.shape[-1]

    def transform(self, X):
        if isinstance(X, dict):
            for n in self.n_features:
                X[n] = self.scalars[n].transform(X[n])
        if isinstance(X, (np.ndarray, np.generic)):
            X = self.scalar.transform(X)
        return X

    def fit_transform(self, X):
        self.fit(X)
        X = self.transform(X)
        return X

def create_directory_safely(path, is_file_path=False):
    try:
        if is_file_path:
            path = os.path.dirname(path)
        print(path)
        if not os.path.exists(path):
            os.mkdir(path)
    except Exception as e:
        print(str(e))

def setup_logging(log_path=None, level=logging.DEBUG):
    """Function setup as many logging for the experiments"""
    if log_path is None:
        dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        dirname = os.path.dirname(dirname)
        log_path = os.path.join(dirname, "logs", "logs.log")
    create_directory_safely(log_path, True)
    logging.basicConfig(filename=log_path, level=level,
                        format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger("SetupLogger")
    logger.info("log file path: {}".format(log_path))
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    # logging.captureWarnings(True)


def setup_random_seed(seed=1234):
    # logger.info('Seed value: {}'.format(seed))
    logger = logging.getLogger("Setup Logging")
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["KERAS_BACKEND"] = "tensorflow"
    devices = tf.config.experimental.list_physical_devices('GPU')
    logger.info("Devices {}".format(devices))
    n_gpus = len(devices)
    logger.info("Number of GPUS {}".format(n_gpus))
    if n_gpus == 0:
        config = ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1,
            allow_soft_placement=True,
            log_device_placement=False,
            device_count={"CPU": multiprocessing.cpu_count() - 2},
        )
    else:
        config = ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
            intra_op_parallelism_threads=2,
            inter_op_parallelism_threads=2,
        )
        config.gpu_options.allow_growth = True
    sess = Session(config=config)
    K.set_session(sess)


def check_file_exists(file_path):
    file_path = os.path.normpath(file_path)
    if not os.path.exists(file_path):
        print("Error: provided file path '%s' does not exist!" % file_path)
        sys.exit(-1)
    return
