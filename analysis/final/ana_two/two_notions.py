# two_notions.py -- POST-HOC ANALYSIS (after the frozen final evaluation; no change to the method): how the two terms of Upright kNN
#   differ as notions of ID conformity on the final evaluation images (seeds 0-2): per-image Spearman correlation of the kNN term and the
#   orientation term on ID images and on each OOD set; per pair the AUROC of each term and of the sum; error overlap at the thresholds
#   that keep 95% of ID images (fraction of OOD images flagged by the kNN term only, the orientation term only, both, neither).
import os, json, numpy as np
from scipy.stats import spearmanr, rankdata
SCR = os.environ.get('UPKNN_ANALYSIS', os.path.expanduser('~/upknn/analysis'))
EXP = f'{SCR}/final/data_kx/export'; COM = json.load(open(f'{SCR}/final/eu_stage/push_final2/final2_idx/final2_idx.json'))
TAGS = ['edmc10', 'l6svhn', 'dpceleba', 'edmc100', 'in64', 'fmnist']
IDN = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'edmc100': 'cifar100_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
def auroc(a, b):
    r = rankdata(np.concatenate([a, b])); return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))
def sub(v, idx, keep):
    pos = {int(i): k for k, i in enumerate(idx)}; return np.asarray([v[pos[int(i)]] for i in keep])
OUT = {}
for t in TAGS:
    sets = [s for s in COM[t] if s != '__id__']
    res = {}
    for sd in (0, 1, 2):
        Z = np.load(f'{EXP}/{t}_s{sd}_final2.npz')
        def get(s, key):
            n = f'final2-{IDN[t] if s == "__id__" else s}'; return sub(Z[f'{key}__{n}'].astype(np.float64), Z[f'idx__{n}'], COM[t][s])
        ci, pi = get('__id__', 'content'), get('__id__', 'pose'); si = ci + pi
        tc, tp, ts = np.quantile(ci, 0.05), np.quantile(pi, 0.05), np.quantile(si, 0.05)       # thresholds keeping 95% of ID (higher = ID)
        res.setdefault('__id__', []).append(dict(rho=spearmanr(ci, pi)[0]))
        for s in sets:
            co, po = get(s, 'content'), get(s, 'pose'); so = co + po
            fc, fp, fs = co < tc, po < tp, so < ts
            res.setdefault(s, []).append(dict(rho=spearmanr(co, po)[0], au_c=100 * auroc(ci, co), au_p=100 * auroc(pi, po), au_s=100 * auroc(si, so),
                                             only_c=100 * np.mean(fc & ~fp), only_p=100 * np.mean(fp & ~fc), both=100 * np.mean(fc & fp), neither=100 * np.mean(~fc & ~fp),
                                             sum_flag=100 * np.mean(fs), miss_c_caught_p=100 * np.mean(fp[~fc]) if (~fc).any() else float('nan')))
    OUT[t] = {s: {k: float(np.nanmean([d[k] for d in v])) for k in v[0]} for s, v in res.items()}
json.dump(OUT, open('two_notions.json', 'w'), indent=1)
for t in TAGS:
    print(f'== {t}: rho(ID) {OUT[t]["__id__"]["rho"]:+.2f}')
    for s, v in OUT[t].items():
        if s == '__id__': continue
        print(f'  {s[:22]:22s} rho {v["rho"]:+.2f} | AUROC knn {v["au_c"]:5.1f} orient {v["au_p"]:5.1f} sum {v["au_s"]:5.1f} | flagged: knn-only {v["only_c"]:4.1f} orient-only {v["only_p"]:4.1f} both {v["both"]:4.1f} neither {v["neither"]:4.1f} | sum {v["sum_flag"]:4.1f} | missed-by-knn caught-by-orient {v["miss_c_caught_p"]:4.1f}')
