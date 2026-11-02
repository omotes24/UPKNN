# ok_common.py -- loaders and components of PROTOCOL_OK.md (cloud side)
import os, json, itertools
import numpy as np
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf
SCR = os.environ.get('UPKNN_ANALYSIS', os.path.expanduser('~/upknn/analysis'))
OKD = f'{SCR}/final/data_ok/sims'              # runs/ok/sims copied from the GPU server
TAGS = ['edmc10', 'l6svhn', 'dpceleba', 'edmc100', 'in64', 'fmnist']
IDN = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'edmc100': 'cifar100_test', 'in64': 'in64_imagenet_test',
       'fmnist': 'g32_fmnist_test'}
ID_NAME = {'edmc10': 'CIFAR-10', 'l6svhn': 'SVHN', 'dpceleba': 'CelebA', 'edmc100': 'CIFAR-100', 'in64': 'ImageNet-64', 'fmnist': 'FashionMNIST'}
PHI = ['g', 'q', 'gq']; HG = ['L0', 'L1', 'L2', 'L012']; KS = [0, 6]; KK = [1, 10]


def load(tag, name, seed=0):
    """{k: dict(S [N,4,3,4,4,2] float32, e [N,4], idx [N])}"""
    out = {}
    for k in KS:
        z = np.load(f'{OKD}/s{seed}/{tag}/k{k}/{name}.npz')
        out[k] = dict(S=z['S'].astype(np.float32), e=z['e'], idx=z['idx'])
    assert (out[0]['idx'] == out[6]['idx']).all()
    return out


def auroc(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    r = rankdata(np.concatenate([a, b]))
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


# ----------------------------------------------------------------------------- components (higher = more ID), passes = set of (view, k)
def comp_value(D, c):
    """D = load(...) ; c = component key tuple"""
    kind = c[0]
    if kind in ('C1', 'C2', 'C4', 'O1', 'O2', 'O4'):
        _, phi, h, k, kk = c
        S = D[k]['S'][:, HG.index(h), PHI.index(phi), :, :, KK.index(kk)]          # [N, 4 r, 4 s]
        diag = np.stack([S[:, r, r] for r in range(4)], 1)
        if kind == 'C1': return S[:, 0, 0]
        if kind == 'C2': return (S[:, 0, 0] + S[:, 2, 2]) / 2
        if kind == 'C4': return diag.mean(1)
        marg = np.stack([S[:, r, r] - np.max(np.delete(S[:, r, :], r, axis=1), 1) for r in range(4)], 1)
        if kind == 'O1': return marg[:, 0]
        if kind == 'O2': return (marg[:, 0] + marg[:, 2]) / 2
        return marg.mean(1)
    if kind == 'A':
        _, r, k = c; e = D[k]['e']; return np.log(e[:, r]) - np.log(e[:, 0])
    if kind == 'T':
        return np.stack([np.log(D[0]['e'][:, 0]), np.log(D[6]['e'][:, 0])], 1)   # vector; distance computed with the calibration fit
    raise ValueError(c)


def comp_passes(c):
    kind = c[0]
    if kind in ('C1', 'O1'): return {(0, c[3])}
    if kind in ('C2', 'O2'): return {(0, c[3]), (2, c[3])}
    if kind in ('C4', 'O4'): return {(r, c[3]) for r in range(4)}
    if kind == 'A': return {(0, c[2]), (c[1], c[2])}
    if kind == 'T': return {(0, 0), (0, 6)}


def logp_low(ref, x):
    r = np.sort(ref); return np.log((1 + np.searchsorted(r, x, 'right')) / (len(r) + 1))


def logp_high(ref, x):
    r = np.sort(ref); return np.log((1 + len(r) - np.searchsorted(r, x, 'left')) / (len(r) + 1))


class Maha:
    def __init__(self, X): self.mu = X.mean(0); self.P = LedoitWolf().fit(X - self.mu).precision_
    def __call__(self, X): Z = X - self.mu; return np.einsum('ij,jk,ik->i', Z, self.P, Z)


def calibrated_logp(c, cal, xs):
    """log p-values (small = OOD) of the arrays xs (list) for component c, calibrated on the array cal"""
    if c[0] == 'T':
        M = Maha(cal); ref = M(cal); return [logp_high(ref, M(x)) for x in xs]
    return [logp_low(cal, x) for x in xs]


def candidates(budget=4):
    out = []
    for kind in ('C1', 'C2', 'C4'):
        for phi, h, k, kk in itertools.product(PHI, HG, KS, KK):
            cont = (kind, phi, h, k, kk)
            oris = [None] + [(o, phi, h, k, kk) for o in ('O1', 'O2', 'O4')] + [('A', r, kp) for r in (1, 2, 3) for kp in KS]
            for o in oris:
                for t in (None, ('T',)):
                    comps = [cont] + ([o] if o else []) + ([t] if t else [])
                    passes = set().union(*[comp_passes(x) for x in comps])
                    if len(passes) <= budget: out.append((tuple(comps), len(passes)))
    return out
