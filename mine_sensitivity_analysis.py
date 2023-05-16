import inspect
import logging
import os

import numpy as np
import wandb
from sklearn.model_selection import StratifiedShuffleSplit

from experiments.utils import *
from pycilt.dataset_readers import SyntheticDatasetGenerator
from pycilt.mi_estimators import MineMIEstimatorHPO

DIR_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
EXPERIMENTS = 'experiments'

if __name__ == "__main__":
    log_path = os.path.join(DIR_PATH, EXPERIMENTS, 'test_mine.log')
    setup_logging(log_path=log_path)
    setup_random_seed(random_state=42)
    logger = logging.getLogger('Experiment')
    script_path = os.path.join(DIR_PATH, 'mine_sensitivity_analysis.py')
    os.environ["WANDB_API_KEY"] = "e1cd10bac622be84198c705e89f0209dd0fc0ac2"
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_NOTEBOOK_NAME"] = script_path

    sweep_configuration = {
        'method': 'random',
        'metric': {'name': 'validation_loss', 'goal': 'minimize'},
        'parameters': {
            'learning_rate': {'min': np.log10(1e-05), 'max': np.log10(0.1), 'distribution': 'uniform'},
            'reg_strength': {'min': np.log10(1e-10), 'max': np.log10(0.1), 'distribution': 'uniform'},
            'optimizer_str': {'values': ['RMSprop', 'sgd', 'adam']},
            'loss_function': {'values': ['donsker_varadhan_softplus', 'donsker_varadhan', 'fdivergence']},
            'n_units': {'values': list(range(10, 257))},
            'n_hidden': {'values': list(range(1, 6))},
            # 'encode_classes': {'values': [True, False]}
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='mine-sweep')
    dataset_params = {"n_classes": 5, "n_features": 5, "flip_y": 0.3, 'samples_per_class': 500}
    dataset_reader = SyntheticDatasetGenerator(**dataset_params)
    X, y = dataset_reader.generate_dataset()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=0)
    train_index, test_index = list(sss.split(X, y))[0]
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    actual_mi = dataset_reader.bayes_predictor_mi()
    logger.info(f"Actual MI {actual_mi}")


    def main():
        run = wandb.init()

        # note that we define values from `wandb.config` instead
        # of defining hard values
        learning_rate = 10 ** wandb.config.learning_rate
        reg_strength = 10 ** wandb.config.reg_strength
        optimizer_str = wandb.config.optimizer_str
        loss_function = wandb.config.loss_function
        n_units = wandb.config.n_units
        n_hidden = wandb.config.n_hidden
        wandb.config.actual_learning_rate = learning_rate
        wandb.config.actual_reg_strength = reg_strength
        # encode_classes = wandb.config.encode_classes
        wandb.config.update(d={
            'actual_learning_rate': learning_rate,
            'actual_reg_strength': reg_strength,
        })

        n_features = X_train.shape[-1]
        n_classes = len(np.unique(y_train))

        params = {'n_classes': n_classes, 'n_features': n_features, 'n_units': n_units, 'n_hidden': n_hidden,
                  'learning_rate': learning_rate, 'reg_strength': reg_strength, 'optimizer_str': optimizer_str,
                  'encode_classes': True, 'loss_function': loss_function}
        clf = MineMIEstimatorHPO(**params)
        clf.fit(X_train, y_train, epochs=10000, verbose=0)
        train_mi = clf.estimate_mi(X_train, y_train)
        mi = clf.estimate_mi(X_test, y_test)
        val_loss = np.abs((actual_mi - mi))
        actual_loss = np.abs((actual_mi - train_mi))
        train_mse = clf.score(X_train, y_train)
        val_mse = clf.score(X_test, y_test)
        wandb.log({"val mi": mi, "train mi": train_mi})
        wandb.log({"val_mse": val_mse, "train_mse": train_mse})
        wandb.log({"actual_loss": actual_loss, "validation_loss": val_loss})
        logger.info(f"params: {params}")
        logger.info(f"actual_loss: {actual_loss}, val_loss: {val_loss}")
        logger.info(f"actual_mi: {actual_mi} val mi: {mi}, train mi: {train_mi}")
        logger.info(f"train_mse: {train_mse} val_mse: {val_mse}")

    # Start sweep job.
    wandb.agent(sweep_id, function=main, count=500)
