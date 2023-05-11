from autogluon.core.space import Real, Int

hyperparameters = {
    "NN_TORCH": {
        "learning_rate": Real(1e-5, 1e-1, default=5e-4, log=True),
        "dropout_prob": Real(0.0, 0.5, default=0.1),
        "num_layers": Int(lower=2, upper=20, default=5),
        "hidden_size": Int(lower=8, upper=256, default=32)
    },
    "GBM": {
        "n_estimators": Int(20, 300),
        "learning_rate": Real(1e-2, 0.3, log=True),
        "max_depth": Int(3, 20),
        "num_leaves": Int(20, 300),
        'feature_fraction': Real(0.2, 0.95, log=True),
        'bagging_fraction': Real(0.2, 0.95, log=True),
        'min_data_in_leaf': Int(20, 5000),
        'lambda_l1': Real(1e-6, 1e-2, log=True),
        'lambda_l2': Real(1e-6, 1e-2, log=True),
    },
    "CAT": {
        "learning_rate": Real(1e-2, 0.5, log=True),
        'depth': Int(4, 10),
        'l2_leaf_reg': Real(0.1, 10)
    },
    "XGB": {
        "n_estimators": Int(20, 300),
        "max_depth": Int(3, 10),
        "learning_rate": Real(0.1, 0.5, log=True),
        "gamma": Real(0.0, 1.0),
        "reg_alpha": Real(0.0, 1.0),
        "reg_lambda": Real(0.0, 1.0)
    },
    "FASTAI": {
        "learning_rate": Real(1e-5, 1e-1, default=5e-4, log=True),
        'wd': Real(0.0, 0.5),
        'emb_drop': Real(0.0, 0.5),
    },
    "RF": { #Note: Hyperparameter tuning is disabled for this model by autogluon
        # "n_estimators": Int(20, 300),
        # "criterion": Categorical("gini", "entropy"),
        # "max_depth": Int(lower=6, upper=20, default=10),
        # "max_features": Categorical("sqrt", "log2"),
        # "min_samples_leaf": Int(lower=2, upper=50, default=10),
        # "min_samples_split": Int(lower=2, upper=50, default=10)
    },
    "XT": { #Note: Hyperparameter tuning is disabled for this model by autogluon
        #"n_estimators": Int(20, 300),
        #"criterion": Categorical("gini", "entropy"),
        #"max_depth": Int(lower=6, upper=20, default=10),
        #"max_features": Categorical("sqrt", "log2"),
        #"min_samples_leaf": Int(lower=2, upper=50, default=10),
        #"min_samples_split": Int(lower=2, upper=50, default=10)
    },
    "KNN": { #Note: Hyperparameter tuning is disabled for this model by autogluon
        #"weights": Categorical("uniform", "distance"),
        #"n_neighbors": Int(lower=3, upper=10, default=5),
        #"p": Categorical(1, 2, 3)
    }
}