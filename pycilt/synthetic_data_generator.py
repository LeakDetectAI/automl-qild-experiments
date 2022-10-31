import numpy as np
from abc import ABCMeta
from scipy.stats import multivariate_normal
from scipy.stats import ortho_group
from sklearn.utils import check_random_state, shuffle


def pdf(dist, x):
    return np.exp(dist.logpdf(x))


class SyntheticDatasetGenerator(metaclass=ABCMeta):
    def __init__(self, n_classes=2, n_features=2, samples_per_class=1000, flip_y=0.1, random_state=42, fold_id=0):
        self.n_classes = n_classes
        self.n_features = n_features
        self.random_state = check_random_state(random_state)
        self.fold_id = fold_id
        self.means = {}
        self.covariances = {}
        self.seeds = {}
        if isinstance(samples_per_class, int):
            self.samples_per_class = dict.fromkeys(np.arange(n_classes), samples_per_class)
        elif isinstance(samples_per_class, dict):
            self.samples_per_class = samples_per_class
        else:
            raise ValueError("Samples per class is not defined properly")
        self.n_instances = sum(self.samples_per_class.values())
        self.class_labels = np.arange(self.n_classes)
        self.y_prob = {}
        self.flip_y_prob = {}
        self.flip_y = flip_y
        self.generate_cov_means()

    def generate_cov_means(self):
        for k_class in self.class_labels:
            # A = rs.rand(n_features, n_features)
            # matrix1 = np.matmul(A, A.transpose())
            seed = self.random_state.randint(2 ** 31, dtype="uint32") + self.fold_id
            rs = np.random.RandomState(seed=seed)
            Q = ortho_group.rvs(dim=self.n_features)
            S = np.diag(np.diag(rs.rand(self.n_features, self.n_features)))
            cov = np.dot(np.dot(Q, S), np.transpose(Q))
            mean = np.ones(self.n_features) + k_class * 1.5
            self.means[k_class] = mean
            self.covariances[k_class] = cov
            self.seeds[k_class] = seed
            self.y_prob[k_class] = self.samples_per_class[k_class] / self.n_instances
            self.flip_y_prob[k_class] = self.y_prob[k_class] * (1 - self.flip_y) + self.flip_y / self.n_classes

    def get_prob_dist_x_given_y(self, k_class):
        return multivariate_normal(mean=self.means[k_class], cov=self.covariances[k_class],
                                   seed=self.seeds[k_class])

    def get_prob_fn_margx(self):
        marg_x = lambda x: np.array(
            [self.y_prob[k_class] * pdf(self.get_prob_dist_x_given_y(k_class), x) for k_class in self.class_labels])
        return marg_x

    def get_prob_x_given_y(self, X, class_label):
        dist = self.get_prob_dist_x_given_y(class_label)
        prob_x_given_y = pdf(dist, X)
        return prob_x_given_y

    def get_prob_y_given_x(self, X, class_label):
        pdf_xy = lambda x, k_class: self.y_prob[k_class] * pdf(self.get_prob_dist_x_given_y(k_class), x)
        marg_x = self.get_prob_fn_margx()
        x_marg = marg_x(X).sum(axis=0)
        prob_y_given_x = pdf_xy(X, class_label) / x_marg
        return prob_y_given_x

    def get_prob_flip_y_given_x(self, X, class_label):
        prob_y_given_x = np.zeros(X.shape[0])
        for k_class in self.class_labels:
            prob_y_given_x += (self.flip_y / self.y_prob[k_class]) * self.get_prob_y_given_x(X, k_class)
            if k_class == class_label:
                prob_y_given_x += (1 - self.flip_y) * self.get_prob_y_given_x(X, k_class)
        return prob_y_given_x

    def get_prob_x_given_flip_y(self, X, class_label):
        prob_flip_y_given_x = self.get_prob_flip_y_given_x(X, class_label)
        marg_x = self.get_prob_fn_margx()
        x_marg = marg_x(X).sum(axis=0)
        prob_x_given_flip_y = (prob_flip_y_given_x * x_marg)
        return prob_x_given_flip_y

    def generate_samples_for_class(self, k_class):
        seed = self.random_state.randint(2 ** 32, dtype="uint32")
        mvn = self.get_prob_dist_x_given_y(k_class)
        n_samples = self.samples_per_class[k_class]
        data = mvn.rvs(n_samples, random_state=seed)
        labels = np.zeros(n_samples) + k_class
        return data, labels

    def generate_dataset(self):
        X = []
        y = []

        for k_class in self.class_labels:
            data, labels = self.generate_samples_for_class(k_class)
            if len(X) == 0:
                X = data
                y = labels
            else:
                X = np.vstack((X, data))
                y = np.append(y, labels)
        if self.flip_y > 0:
            indicies = []
            for k_class in self.class_labels:
                flip_samples = int(self.flip_y * self.samples_per_class[k_class])
                ind0 = list(self.random_state.choice(np.where(y == k_class)[0], flip_samples, replace=False))
                indicies.extend(ind0)
            #y_old = np.copy(y)
            y[indicies] = shuffle(y[indicies], random_state=self.random_state)
        return X, y
