import logging

import numpy as np
import numpy.random as npr
import torch
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

from pycilt.utils import softmax
from .class_nn import StatNet
from .mi_base_class import MIEstimatorBase
from .pytorch_utils import get_optimizer_and_parameters, init, get_mine_loss


class MineMIEstimator(MIEstimatorBase):
    def __init__(self, n_classes, input_dim, n_hidden=2, n_units=100, loss_function='donsker_varadhan_softplus',
                 optimizer_str='adam', learning_rate=0.001, reg_strength=0.001, encode_classes=True, random_state=42):
        super().__init__(n_classes=n_classes, input_dim=input_dim, random_state=random_state)
        self.logger = logging.getLogger(MineMIEstimator.__name__)
        self.optimizer_str = optimizer_str
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.optimizer_cls, self._optimizer_config = get_optimizer_and_parameters(optimizer_str, learning_rate,
                                                                                  reg_strength)
        self.encode_classes = encode_classes
        self.n_hidden = n_hidden
        self.n_units = n_units
        self.loss_function = loss_function
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"device {self.device}")
        print((f"device {self.device}"))
        self.optimizer = None
        self.stat_net = None
        self.dataset_properties = None
        self.label_binarizer = None

    def pytorch_tensor_dataset(self, X, y, i=2):
        seed = self.random_state.randint(2 ** 31, dtype="uint32") + i
        rs = np.random.RandomState(seed)
        if self.encode_classes:
            y_t = self.label_binarizer.transform(y)
            xy = np.hstack((X, y_t))
            y_s = rs.permutation(y)
            y_t = self.label_binarizer.transform(y_s)
            xy_tilde = np.hstack((X, y_t))
        else:
            xy = np.hstack((X, y[:, None]))
            y_s = rs.permutation(y)
            xy_tilde = np.hstack((X, y_s[:, None]))
        tensor_xy = torch.tensor(xy, dtype=torch.float32)  # transform to torch tensor
        tensor_xy_tilde = torch.tensor(xy_tilde, dtype=torch.float32)
        return tensor_xy, tensor_xy_tilde

    def fit(self, X, y, epochs=50000, verbose=0, **kwd):
        MON_FREQ = epochs // 10
        # Monitoring
        MON_ITER = epochs // 50
        if self.encode_classes:
            y_t = LabelBinarizer().fit_transform(y)
            cls_enc = y_t.shape[-1]
        else:
            cls_enc = 1
        self.label_binarizer = LabelBinarizer().fit(y)
        self.stat_net = StatNet(in_dim=self.input_dim, cls_enc=cls_enc, n_hidden=self.n_hidden, n_units=self.n_units)
        self.stat_net.apply(init)
        self.stat_net.to(self.device)
        self.optimizer = self.optimizer_cls(self.stat_net.parameters(), **self._optimizer_config)
        all_estimates = []
        for iter_ in tqdm(range(epochs), total=epochs, desc='iteration'):
            self.stat_net.zero_grad()
            # print(f"iter {iter_}, y {y}")
            xy, xy_tilde = self.pytorch_tensor_dataset(X, y, i=iter_)
            preds_xy = self.stat_net(xy)
            preds_xy_tilde = self.stat_net(xy_tilde)
            train_div = get_mine_loss(preds_xy, preds_xy_tilde, metric=self.loss_function)
            loss = train_div.mul_(-1.)
            loss.backward()
            self.optimizer.step()
            if (iter_ % MON_FREQ == 0) or (iter_ + 1 == epochs):
                with torch.no_grad():
                    mi_hats = []
                    for _ in range(MON_ITER):
                        # print(f"iter {iter_}, y {y}")
                        xy, xy_tilde = self.pytorch_tensor_dataset(X, y, i=iter_)
                        preds_xy = self.stat_net(xy)
                        preds_xy_tilde = self.stat_net(xy_tilde)
                        eval_div = get_mine_loss(preds_xy, preds_xy_tilde, metric=self.loss_function)
                        mi_hats.append(eval_div.cpu().numpy())
                    mi_hat = np.mean(mi_hats)
                    if verbose:
                        print(f'iter: {iter_}, MI hat: {mi_hat}')
                    self.logger.info(f'iter: {iter_}, MI hat: {mi_hat}')
                    all_estimates.append(dict(iter_=iter_, mi_hat=mi_hat))
        return self

    def predict(self, X, verbose=0):
        scores = self.predict_proba(X=X, verbose=verbose)
        y_pred = np.argmax(scores, axis=1)
        return y_pred

    def score(self, X, y, sample_weight=None, verbose=0):
        return self.estimate_mi(X=X, y=y, verbose=verbose)

    def predict_proba(self, X, verbose=0):
        scores = self.decision_function(X=X, verbose=verbose)
        scores = softmax(scores)
        return scores

    def decision_function(self, X, verbose=0):
        scores = None
        for n_class in range(self.n_classes):
            y = np.zeros(X.shape[0]) + n_class
            xy, xy_tilde = self.pytorch_tensor_dataset(X, y, i=0)
            score = self.stat_net(xy).detach().numpy()
            self.logger.info(f"Class {n_class} scores {score.flatten()}")
            if scores is None:
                scores = score
            else:
                scores = np.hstack((scores, score))
        return scores

    def estimate_mi(self, X, y, verbose=0):
        MON_ITER = 1000
        mi_hats = []
        for iter_ in range(MON_ITER):
            xy, xy_tilde = self.pytorch_tensor_dataset(X, y, i=iter_)
            preds_xy = self.stat_net(xy)
            preds_xy_tilde = self.stat_net(xy_tilde)
            eval_div = get_mine_loss(preds_xy, preds_xy_tilde, metric=self.loss_function)
            mi_hat = eval_div.detach().numpy().flatten()[0]
            if verbose:
                print(f'iter: {iter_}, MI hat: {mi_hat}')
            mi_hats.append(mi_hat)
        mi_hats = np.array(mi_hats)
        n = int(MON_ITER / 2)
        mi_hats = mi_hats[np.argpartition(mi_hats, -n)[-n:]]
        mi_hat = np.mean(mi_hats)
        self.logger.info(f'Estimated MIs: {mi_hats} Mean {mi_hat}')
        return mi_hat
