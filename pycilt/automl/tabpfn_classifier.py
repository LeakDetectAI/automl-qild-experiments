import logging

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils import check_random_state
from tabpfn import TabPFNClassifier

from pycilt.automl.automl_core import AutomlClassifier
from ..dimensionality_reduction_techniques import create_dimensionality_reduction_model


class AutoTabPFNClassifier(AutomlClassifier):
    def __init__(self, n_features, n_classes, n_ensembles=100, n_reduced=20, reduction_technique='select_from_model_rf',
                 random_state=None, **kwargs):
        self.n_features = n_features
        self.n_classes = n_classes
        self.logger = logging.getLogger(name=AutoTabPFNClassifier.__name__)
        self.random_state = check_random_state(random_state)

        self.n_reduced = n_reduced
        self.reduction_technique = reduction_technique
        self.selection_model = None
        self.__is_fitted__ = False

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = device
        self.logger.info(f"Device {self.device}")
        self.n_ensembles = n_ensembles
        self.model = None

    def transform(self, X, y=None):
        self.logger.info(f"Before transform n_features {X.shape[-1]}")
        if y is not None:
            classes, n_classes = np.unique(y, return_counts=True)
            self.logger.info(f"Classes {classes} No of Classes {n_classes}")
        if not self.__is_fitted__:
            if self.n_features != X.shape[-1]:
                raise ValueError(f"Dataset passed does not contain {self.n_features}")
            if y is not None:
                if self.n_classes != len(np.unique(y)):
                    raise ValueError(f"Dataset passed does not contain {self.n_classes}")
            self.selection_model = create_dimensionality_reduction_model(reduction_technique=self.reduction_technique,
                                                                         n_reduced=self.n_reduced)
            self.logger.info(f"Creating the model")
            if self.n_features > 50 and self.n_reduced < self.n_features:
                self.logger.info(f"Transforming and reducing the {self.n_features} features to {self.n_reduced}")
                self.selection_model.fit(X, y)
                X = self.selection_model.transform(X)
                self.__is_fitted__ = True
        else:
            if self.n_features > 50 and self.n_reduced < self.n_features:
                X = self.selection_model.transform(X)
        self.logger.info(f"After transform n_features {X.shape[-1]}")
        return X

    def fit(self, X, y, **kwd):
        X = self.transform(X, y)
        self.model = TabPFNClassifier(device=self.device, N_ensemble_configurations=self.n_ensembles)
        self.model.fit(X, y, overwrite_warning=True)
        self.clear_memory()
        self.logger.info("Fitting Done")


    def predict(self, X, verbose=0):
        p = self.predict_proba(X, verbose=0)
        y_pred = np.argmax(p, axis=-1)
        self.logger.info("Predict Done")
        return y_pred

    def score(self, X, y, sample_weight=None, verbose=0):
        y_pred = self.predict(X)
        acc = balanced_accuracy_score(y, y_pred)
        return acc

    def predict_proba(self, X, batch_size=200, verbose=0):
        n_samples = X.shape[0]
        n_batches = np.ceil(n_samples / batch_size).astype(int)
        predictions = []
        if self.device == "cuda":
            batch_size = 50
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            X_batch = X[start_idx:end_idx]
            X_transformed = self.transform(X_batch)
            self.logger.info(f"Processing batch {i + 1}/{n_batches} Start id {start_idx} end id {end_idx}")
            batch_pred = self.model.predict_proba(X_transformed, normalize_with_test=True, return_logits=False)
            predictions.append(batch_pred)

        y_pred = np.concatenate(predictions, axis=0)
        self.logger.info("Predict_proba Done")
        self.clear_memory()
        return y_pred

    def decision_function(self, X, verbose=0):
        return self.predict_proba(X, verbose)

    def clear_memory(self):
        # Call Python's garbage collector
        import gc
        gc.collect()
        # Explicitly clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

