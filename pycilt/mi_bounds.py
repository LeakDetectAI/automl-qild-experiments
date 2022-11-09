import numpy as np
from sklearn.metrics import accuracy_score

__all__ = ['bin_ce', 'helmann_raviv_function', 'helmann_raviv_upper_bound', 'santhi_vardi_upper_bound',
           'fanos_lower_bound', 'fanos_adjusted_lower_bound']


def bin_ce(p_e):
    p_e = p_e + 1e-200
    be = -p_e * np.log2(p_e) - (1 - p_e) * np.log2(1 - p_e)
    return be


bce_f = np.vectorize(bin_ce)


def helmann_raviv_function(n_classes, pe):
    ls = []
    indicies = []
    num = pe.shape[0]
    for k in range(1, int(n_classes)):
        def cal_l(k, n_pe):
            T = (k + 1) / k
            T2 = (k - 1) / k
            l = np.log2(k) + k * (k + 1) * np.log2(T) * (n_pe - T2)
            # T = (k+1)/k
            # T2 = (1+1/k)
            # l = np.log2(k+1) + k**2*np.log2(T)*(pe*T2-1)
            return l

        l_mpe = (1 - 1 / k)
        u_mpe = (1 - 1 / (k + 1))
        idx = np.where((pe >= l_mpe) & (pe < u_mpe))[0]
        indicies.extend(idx)
        if len(idx) != 0:
            n_pe = pe[idx]
            l = cal_l(k, n_pe)
            ls.extend(l)
        # else:
        #   print(k, l_mpe, u_mpe, mpe)

        # plt.plot(pe, l, label='Hellman-Raviv k={}-{}'.format(k ,LOWER), linewidth=2, color='tab:orange')
    idx = np.array(list(set(np.arange(num)) ^ set(indicies)))
    if len(idx) != 0:
        n_pe = pe[idx]
        l = cal_l(k, n_pe)
        ls.extend(l)
    ls = np.array(ls)
    return ls


def helmann_raviv_upper_bound(y_true, y_pred):
    n_classes = len(np.unique(y_true))
    acc = accuracy_score(y_true, y_pred)
    pe = 1 - acc
    hmr = helmann_raviv_function(n_classes, np.array([pe]))[0]
    u = np.log2(n_classes) - hmr
    return u


def santhi_vardi_upper_bound(y_true, y_pred):
    n_classes = len(np.unique(y_true))
    acc = accuracy_score(y_true, y_pred)
    pe = 1 - acc
    u = np.log2(n_classes) + np.log2(1 - pe)
    return u


def fanos_lower_bound(y_true, y_pred):
    n_classes = len(np.unique(y_true))
    acc = accuracy_score(y_true, y_pred)
    pe = 1 - acc
    T = np.log(n_classes - 1) / np.log(n_classes)

    l = np.log2(n_classes) * (1 - pe * T) - bin_ce(pe)
    return l


def fanos_adjusted_lower_bound(y_true, y_pred):
    n_classes = len(np.unique(y_true))
    acc = accuracy_score(y_true, y_pred)
    pe = 1 - acc
    l = np.log2(n_classes) * (1 - pe) - bce_f(pe)
    return l
