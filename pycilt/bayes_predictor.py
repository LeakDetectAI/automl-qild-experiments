import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state


class BayesPredictor(BaseEstimator, ClassifierMixin):

    def __init__(self, dataset_obj, random_state=None, **kwargs):
        self.dataset_obj = dataset_obj
        self.random_state = check_random_state(random_state)


    def fit(self, X, y, **kwd):
        pass

    def predict(self, X, verbose=0):
        pred_probabilities = self.predict_proba(X=X, verbose=verbose)
        y_pred = pred_probabilities.argmax(axis=1)
        return y_pred

    def score(self, X, y, sample_weight=None, verbose=0):
        y_pred = self.predict(X)
        acc_bp = np.mean(y_pred == y)
        return acc_bp

    def predict_proba(self, X, verbose=0):
        prob_predictions = np.zeros((X.shape[0], self.dataset_obj.n_classes))
        for k_class in self.dataset_obj.class_labels:
            if self.dataset_obj.flip_y == 0.0:
                prob_predictions[:, k_class] = self.dataset_obj.get_prob_y_given_x(X=X, class_label=k_class)
            else:
                prob_predictions[:, k_class] = self.dataset_obj.get_prob_flip_y_given_x(X=X, class_label=k_class)
        return prob_predictions

    def decision_function(self, X, verbose=0):
        prob_predictions = self.predict_proba(X=X, verbose=verbose)
        return prob_predictions
