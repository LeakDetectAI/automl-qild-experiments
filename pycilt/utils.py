import sys
import traceback
import warnings

import numpy as np
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

__all__ = ['logsumexp', 'softmax', 'progress_bar', 'print_dictionary', 'standardize_features', 'standardize_features']


def logsumexp(x, axis=1):
    max_x = x.max(axis=axis, keepdims=True)
    return max_x + np.log(np.sum(np.exp(x - max_x), axis=axis, keepdims=True))


def softmax(x, axis=1):
    """
    Take softmax for the given numpy array.
    :param axis: The axis around which the softmax is applied
    :param x: array-like, shape (n_samples, ...)
    :return: softmax taken around the given axis
    """
    lse = logsumexp(x, axis=axis)
    return np.exp(x - lse)

def sigmoid(x):
    x = 1.0 / (1.0 + np.exp(-x))
    return x

def normalize(x, axis=1):
    """
    Normalize the given two dimensional numpy array around the row.
    :param axis: The axis around which the norm is applied
    :param x: theano or numpy array-like, shape (n_samples, n_objects)
    :return: normalize the array around the axis=1
    """
    return x / np.sum(x, axis=axis, keepdims=True)

def progress_bar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s/%s ...%s\r' % (bar, count, total, status))
    sys.stdout.flush()


def print_dictionary(dictionary, sep='\n'):
    output = "  "
    for i, (key, value) in enumerate(dictionary.items()):
        if i < len(dictionary) - 1:
            output = output + str(key) + " => " + str(value) + sep
        else:
            output = output + str(key) + " => " + str(value)
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


def log_exception_error(logger, e):
    if hasattr(e, 'message'):
        message = e.message
    else:
        message = e
    logger.error(traceback.format_exc())
    logger.error(message)


