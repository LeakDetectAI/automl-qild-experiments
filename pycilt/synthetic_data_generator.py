import logging
from abc import ABCMeta

import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import ortho_group
from sklearn.utils import check_random_state, shuffle


def pdf(dist, x):
    return np.exp(dist.logpdf(x))


class SyntheticDatasetGenerator(metaclass=ABCMeta):
    def __init__(self, n_classes=2, n_features=2, samples_per_class=500, flip_y=0.1, random_state=42, fold_id=0):
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
        self.logger = logging.getLogger(SyntheticDatasetGenerator.__name__)

    def generate_cov_means(self):
        # seed = self.random_state.randint(2 ** 32, dtype="uint32")
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
        # print(self.y_prob)
        # print(self.flip_y_prob)

    def get_prob_dist_x_given_y(self, k_class):
        return multivariate_normal(mean=self.means[k_class], cov=self.covariances[k_class],
                                   seed=self.seeds[k_class])

    def get_prob_fn_margx(self):
        marg_x = lambda x: np.array([self.y_prob[k_class] * pdf(self.get_prob_dist_x_given_y(k_class), x)
                                     for k_class in self.class_labels])
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
        prob_y_given_x = (1 - self.flip_y) * self.get_prob_y_given_x(X, class_label) + (
                    self.flip_y * (1 / self.n_classes))
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
            y_old = np.copy(y)
            indicies = []
            for k_class in self.class_labels:
                flip_samples = int(self.flip_y * self.samples_per_class[k_class])
                ind0 = list(self.random_state.choice(np.where(y == k_class)[0], flip_samples, replace=False))
                indicies.extend(ind0)
            # print(f"Orignal {np.unique(y_old, return_counts=True)}")
            y_old[indicies] = shuffle(y[indicies], random_state=self.random_state)
            # print(f"Shuffle {np.unique(y_old, return_counts=True)}")
            if len(np.unique(list(self.samples_per_class.values()))) > 1:
                # print(f"Orignal {np.unique(y, return_counts=True)}")
                choices = []
                indicies = []
                p = [1 / self.n_classes for i in range(self.n_classes)]
                k_classes = list(np.arange(self.n_classes))
                for i, y_i in enumerate(y):
                    choice = self.random_state.choice(2, 1, p=[1 - self.flip_y, self.flip_y])
                    choices.append(choice)
                    if choice == 1:
                        indicies.append(i)
                        # y[i] = self.random_state.choice(self.n_classes, 1)
                y[indicies] = self.random_state.randint(self.n_classes, size=len(indicies))
                self.logger.info(f"np.mean(choices) {np.mean(choices)}")
                uni, counts = np.unique(y, return_counts=True)
                self.logger.info(f"Flipping {uni, counts}")
                self.logger.info(f"Flipping Ratio {uni, counts/np.sum(counts)}")
            else:
                y = y_old
        return X, y

    def calculate_mi(self):
        x_y_prob_list = []
        for k_class in self.class_labels:
            prob_list = -1
            nter = 0
            while prob_list < 0:
                X, y = self.generate_dataset()
                ind = np.where(y == k_class)[0]
                data = X[ind, :]
                if self.flip_y == 0.0:
                    x_y_prob = self.get_prob_x_given_y(X=data, class_label=k_class)
                    marg_x = self.get_prob_fn_margx()
                    p_x_marg = marg_x(data).sum(axis=0)
                    a_log_x_prob = (x_y_prob / p_x_marg)
                else:
                    x_y_prob = self.get_prob_flip_y_given_x(X=data, class_label=k_class)
                    a_log_x_prob = (x_y_prob / self.flip_y_prob[k_class])
                    # print(x_y_prob)
                prob_list = np.nanmean(np.log(a_log_x_prob))
                # print(prob_list)
                nter = nter + 1
                if nter >= 100:
                    break
            if prob_list < 0:
                prob_list = -1 * prob_list
            if self.flip_y == 0.0:
                x_y_prob_list.append(prob_list * self.y_prob[k_class])
            else:
                x_y_prob_list.append(prob_list * self.flip_y_prob[k_class])
        mi = np.nansum(x_y_prob_list)
        return mi

    def bayes_predictor_mi(self):
        X, y = self.generate_dataset()
        y_pred = np.zeros((X.shape[0], self.n_classes))
        for k_class in self.class_labels:
            if self.flip_y == 0.0:
                y_pred[:, k_class] = self.get_prob_y_given_x(X=X, class_label=k_class)
            else:
                y_pred[:, k_class] = self.get_prob_flip_y_given_x(X=X, class_label=k_class)
        marg_x = self.get_prob_fn_margx()
        p_x_marg = marg_x(X).sum(axis=0)
        y_pred[y_pred == 0] = np.finfo(float).eps
        y_pred[y_pred == 1] = 1 - np.finfo(float).eps
        pyx = (y_pred * np.log2(y_pred)).sum(axis=1)
        if self.flip_y < 0.2:
            mi_bp = (pyx * p_x_marg).mean()
        else:
            mi_bp = (pyx).mean()
        mi_pp = 0
        for k_class in self.class_labels:
            mi_pp += -self.flip_y_prob[k_class] * np.log2(self.flip_y_prob[k_class])
        mi = mi_bp + mi_pp
        return mi

    def get_estimated_mi_bounds(self):
        mi = 0
        joint = 0
        for kclass1 in range(self.n_classes):
            mvn = self.get_prob_dist_x_given_y(kclass1)
            # print(self.y_prob[i]*mvn.entropy())
            mi += self.y_prob[kclass1] * mvn.entropy()
            # print(self.y_prob[i]*(-np.log(self.y_prob[i]) + mvn.entropy()))

            joint += -self.y_prob[kclass1] * np.log(self.y_prob[kclass1]) + self.y_prob[kclass1] * mvn.entropy()
        mi_upper = joint - mi
        print(f"MI upper bound H(X|Y) {mi} H(x) {joint} MI {mi_upper}")

        joint = []
        mi = 0
        for kclass1 in range(self.n_classes):
            w_i = self.y_prob[kclass1]
            mvn = self.get_prob_dist_x_given_y(kclass1)
            mi += self.y_prob[kclass1] * mvn.entropy()
            joint_mi = 0
            for kclass2 in range(self.n_classes):
                mvn2 = self.get_prob_dist_x_given_y(kclass2)
                w_j = self.y_prob[kclass2]

                def integrand(*args):
                    input_arr = [x for x in args]
                    p1 = pdf(mvn, input_arr)
                    p2 = pdf(mvn2, input_arr)
                    return p1 * p2

                ranges = [[-np.inf, np.inf] for i in range(self.n_features)]
                # integration = integrate.nquad(integrand, ranges)
                value = multivariate_normal(mean=self.means[kclass2],
                                            cov=self.covariances[kclass1] + self.covariances[kclass2],
                                            seed=self.seeds[kclass1] + self.seeds[kclass2]).pdf(self.means[kclass1])
                # print(f"Overlap between MVNs {kclass1} {kclass2} Integration {integration[0]} Value {value}")
                joint_mi += w_j * value
                # if kclass1 == kclass2:
                # joint_mi+=w_j*value
                # else:
                # joint_mi+=w_j*integration[0]
                # joint_mi+=w_j*integrate.romberg(integrand, ranges)[0]
            # print(f"Classs {kclass1} Joint MI {joint_mi}")
            joint.append(-w_i * np.log(joint_mi))
        mi_lower = np.sum(joint) - mi
        if mi_lower > mi_upper:
            mi_lower = mi_upper
        print(f"MI lower bound H(X|Y) {mi} H(x) {np.sum(joint)} MI {np.sum(joint) - mi}")
        return mi_upper, mi_lower
