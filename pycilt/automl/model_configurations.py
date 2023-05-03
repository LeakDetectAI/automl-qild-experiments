from autogluon.core.space import Categorical, Real, Int

hyperparameters = {
    "NN_TORCH": {
        "learning_rate": Real(1e-5, 1e-2, default=5e-4, log=True),
        "dropout_prob": Real(0.0, 0.5, default=0.1),
        "num_layers": Int(lower=2, upper=20, default=5),
        "hidden_size": Int(lower=8, upper=256, default=32),
    },
    "GBM": {
        "num_boost_round": Int(50, 500),
        "learning_rate": Real(1e-5, 0.5, log=True),
        "max_depth": Int(3, 20),
        "gamma": Real(1e-5, 1.0),
        "alpha": Real(1e-5, 1.0),
        "lambda": Real(0.0, 1.0),
    },
    "CAT": {
        "learning_rate": Real(1e-5, 0.5, log=True),
        'depth': Int(4, 10),
        'l2_leaf_reg': Real(0.1, 10)
    },
    "XGB": {
        "n_estimators": Int(20, 500),
        "max_depth": Int(3, 10),
        "learning_rate": Real(1e-5, 0.5, log=True),
        "gamma": Real(1e-5, 1.0),
        "alpha": Real(1e-5, 1.0),
        "lambda": Real(0.0, 1.0),
    },
    "FASTAI": {
        "learning_rate": Real(1e-5, 1e-2, default=5e-4, log=True),
        'early_stopping': Categorical(True, False),
        'early_stopping_patience': Int(5, 20),
        'weight_decay': Real(0.0, 0.1),
        'momentum': Real(0.8, 0.99),
        'opt_func': Categorical('Adam', 'SGD'),
    },
    "RF": {
        "n_estimators": Int(20, 500),
        "criterion": Categorical("gini", "entropy"),
        "max_depth": Int(lower=6, upper=20, default=10),
        "max_features": Categorical("sqrt", "log2"),
        "min_samples_leaf": Int(lower=2, upper=50, default=10),
        "min_samples_split": Int(lower=2, upper=50, default=10),
    },
    "XT": {
        "n_estimators": Int(20, 500),
        "criterion": Categorical("gini", "entropy"),
        "max_depth": Int(lower=6, upper=20, default=10),
        "max_features": Categorical("sqrt", "log2"),
        "min_samples_leaf": Int(lower=2, upper=50, default=10),
        "min_samples_split": Int(lower=2, upper=50, default=10),
    },
    "KNN": {
        "weights": Categorical("uniform", "distance"),
        "n_neighbors": Int(lower=3, upper=10, default=5),
        "p": Categorical(1, 2, 3),
    },
}