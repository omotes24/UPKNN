# boot.py -- weighted (bootstrap) AUROC without re-sorting: AUROC_r = sum_i w_ri [sum_{b_j < a_i} v_rj + 1/2 sum_{b_j = a_i} v_rj] / (n m)
import numpy as np


def counts(rng, n, B):
    """multinomial bootstrap counts [B, n] (resampling n items with replacement)"""
    idx = rng.integers(0, n, size=(B, n))
    W = np.zeros((B, n), np.float32)
    np.add.at(W, (np.repeat(np.arange(B), n), idx.ravel()), 1.0)
    return W


def boot_auroc(a, b, Wa, Wb):
    """a [n], b [m] scores (higher = ID); Wa [B, n], Wb [B, m] counts -> [B] AUROCs (in %)"""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    ob = np.argsort(b, kind='mergesort'); bs = b[ob]
    pl = np.searchsorted(bs, a, 'left'); pr = np.searchsorted(bs, a, 'right')
    C = np.concatenate([np.zeros((Wb.shape[0], 1), np.float64), np.cumsum(Wb[:, ob], 1, dtype=np.float64)], 1)    # [B, m+1]
    less = C[:, pl]; tie = C[:, pr] - less
    num = (Wa * (less + 0.5 * tie)).sum(1)
    return 100.0 * num / (Wa.sum(1) * Wb.sum(1))


if __name__ == '__main__':
    from scipy.stats import rankdata
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 1, 300).round(1); b = rng.normal(0, 1, 200).round(1)
    Wa = counts(rng, 300, 5); Wb = counts(rng, 200, 5)
    got = boot_auroc(a, b, Wa, Wb)
    ref = []
    for r in range(5):
        aa = np.repeat(a, Wa[r].astype(int)); bb = np.repeat(b, Wb[r].astype(int))
        rk = rankdata(np.concatenate([aa, bb])); ref.append(100 * (rk[:len(aa)].sum() - len(aa) * (len(aa) + 1) / 2) / (len(aa) * len(bb)))
    print(np.round(got, 6), np.round(ref, 6))
    one = boot_auroc(a, b, np.ones((1, 300)), np.ones((1, 200)))
    rk = rankdata(np.concatenate([a, b])); print(one, 100 * (rk[:300].sum() - 300 * 301 / 2) / (300 * 200))
