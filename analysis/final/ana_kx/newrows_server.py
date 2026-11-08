# newrows_server.py -- newrows_final2.py run on the GPU server, where every input already is (no change of method: same rows,
#   same images, same bootstrap draws; the result is checked against the RotCls4 intervals of ours_vs_all_final2.json exactly
#   as the original).  Inputs under $UPKNN_ROOT (default ~/upknn):
#     eu/runs/jobs/f2_idx/<tag>__<set>.json   the final evaluation images of every set (the common images of the paper)
#     runs/kx/export/<tag>_s<seed>_final2.npz  Upright kNN per image (seeds 0-2)
#     runs/rotcls/<tag>/scores_final2/<set>.npz   rotation classifier (bootstrap check only)
#     runs/nll/F_<tag>_final2/<set>.npz        PF-ODE log-likelihood per image (the NLL row of the main evaluation)
#     runs/graham, runs/lmd, runs/kamkari      the three further comparison methods (<tag>_f2)
#   usage: UPKNN_ROOT=<root of the runs> python newrows_server.py <ours_vs_all_final2.json> <out.json>
#   Rows are written only for tiers whose runs are complete (all sets present), so the same command fills the table in later.
import os, sys, json, time
import numpy as np
from scipy.stats import rankdata
ROOT = os.environ.get('UPKNN_ROOT', os.path.expanduser('~/upknn')); SPLIT = 'final2'
RGB = ['cifar10_test', 'cifar100_test', 'svhn_test', 'celeba_test', 'textures', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val']
TIERS = {
    'edmc10':   dict(id='cifar10_test', ood=[s for s in RGB if s != 'cifar10_test'] + ['cifar10_test__vflip', 'synth_noise', 'synth_const']),
    'l6svhn':   dict(id='svhn_test', ood=[s for s in RGB if s != 'svhn_test'] + ['svhn_test__vflip', 'synth_noise', 'synth_const']),
    'dpceleba': dict(id='celeba_test', ood=[s for s in RGB if s != 'celeba_test'] + ['celeba_test__vflip', 'synth_noise', 'synth_const']),
    'edmc100':  dict(id='cifar100_test', ood=[s for s in RGB if s != 'cifar100_test'] + ['cifar100_test__vflip', 'synth_noise', 'synth_const']),
    'in64':     dict(id='in64_imagenet_test', ood=['in64_ssb_hard', 'in64_ninco', 'in64_inaturalist', 'in64_textures', 'in64_openimage_o', 'in64_places365', 'in64_celeba_test', 'in64_imagenet_test__vflip']),
    'fmnist':   dict(id='g32_fmnist_test', ood=['g32_mnist_test', 'g32_kmnist_test', 'g32_omniglot_eval', 'g32_notmnist', 'g32_fmnist_test__vflip', 'g32_synth_noise', 'g32_synth_const']),
}
NAT = {t: [s for s in T['ood'] if not s.endswith('__vflip') and 'synth' not in s] for t, T in TIERS.items()}
ID_NAME = {'edmc10': 'CIFAR-10', 'l6svhn': 'SVHN', 'dpceleba': 'CelebA', 'edmc100': 'CIFAR-100', 'in64': 'ImageNet-64', 'fmnist': 'FashionMNIST'}
B = 2000; SEEDS = (0, 1, 2)
LPCA_PUB = {'edmc10': 193.0, 'l6svhn': 60.0, 'edmc100': 165.0, 'dpceleba': 169.0, 'fmnist': 222.48}   # dgm_ood_detection configs / dgm_geometry
TRAIN = {'edmc10': 'cifar10_train', 'l6svhn': 'svhn_train', 'dpceleba': 'celeba_train', 'edmc100': 'cifar100_train', 'fmnist': 'g32_fmnist_train'}
REF = json.load(open(sys.argv[1])); OUTPATH = sys.argv[2]


def nm(tag, s): return TIERS[tag]['id'] if s == '__id__' else s


COMX = {tag: {s: json.load(open(f'{ROOT}/eu/runs/jobs/f2_idx/{tag}__{nm(tag, s)}.json')) for s in ['__id__'] + T['ood']} for tag, T in TIERS.items()}


def sub(score, idx, keep):
    """the entries of score (indexed like idx) for the image indices in keep, in the order of keep (analysis/common.py)"""
    pos = {int(i): j for j, i in enumerate(np.asarray(idx).tolist())}
    return np.asarray(score)[[pos[i] for i in keep if i in pos]]


def auroc(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    r = rankdata(np.concatenate([a, b]))
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def fpr95(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    thr = np.percentile(a[np.isfinite(a)], 5)
    return float(np.mean(b[np.isfinite(b)] >= thr))


# ---- boot.py (unchanged)
def counts(rng, n, B):
    idx = rng.integers(0, n, size=(B, n))
    W = np.zeros((B, n), np.float32)
    np.add.at(W, (np.repeat(np.arange(B), n), idx.ravel()), 1.0)
    return W


def boot_auroc(a, b, Wa, Wb):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    ob = np.argsort(b, kind='mergesort'); bs = b[ob]
    pl = np.searchsorted(bs, a, 'left'); pr = np.searchsorted(bs, a, 'right')
    C = np.concatenate([np.zeros((Wb.shape[0], 1), np.float64), np.cumsum(Wb[:, ob], 1, dtype=np.float64)], 1)
    less = C[:, pl]; tie = C[:, pr] - less
    num = (Wa * (less + 0.5 * tie)).sum(1)
    return 100.0 * num / (Wa.sum(1) * Wb.sum(1))


def verdict(lo, hi, gap):
    if abs(gap) < 0.05: return 'tie'
    return 'win' if lo > 0 else ('loss' if hi < 0 else 'tie')


def ours(tag, s, seed):
    Z = np.load(f'{ROOT}/runs/kx/export/{tag}_s{seed}_{SPLIT}.npz'); n = f'{SPLIT}-{nm(tag, s)}'
    return sub(Z[f'score__{n}'].astype(np.float64), Z[f'idx__{n}'], COMX[tag][s])


def rotcls4(tag, s):
    z = np.load(f'{ROOT}/runs/rotcls/{tag}/scores_{SPLIT}/{nm(tag, s)}.npz'); return sub(z['rot4'].astype(np.float64), z['idx'], COMX[tag][s])


def nll(tag, s):
    z = np.load(f'{ROOT}/runs/nll/F_{tag}_{SPLIT}/{nm(tag, s)}.npz'); return sub(z['ll'].astype(np.float64), z['idx'], COMX[tag][s])


# ---- Graham: z-score statistics from the __dev128 run of the tier (ID development images, outside the evaluation images)
def graham_stats(tag):
    p = f'{ROOT}/runs/graham/{tag}_f2/{TIERS[tag]["id"]}__dev128.npz'
    if not os.path.exists(p): return None
    zd = np.load(p); return (zd['mse'].mean(0), zd['lpips'].mean(0)), (zd['mse'].std(0) + 1e-12, zd['lpips'].std(0) + 1e-12)


def graham(tag, s, stats):
    p = f'{ROOT}/runs/graham/{tag}_f2/{nm(tag, s)}.npz'
    if not os.path.exists(p) or stats is None: return None
    z = np.load(p); (mu, sd) = stats
    v = -np.concatenate([(z['mse'] - mu[0]) / sd[0], (z['lpips'] - mu[1]) / sd[1]], 1).mean(1)     # higher = ID
    return sub(v.astype(np.float64), z['idx'], COMX[tag][s])


def lmd(tag, s):
    p = f'{ROOT}/runs/lmd/{tag}_f2/{nm(tag, s)}.npz'
    if not os.path.exists(p): return None
    z = np.load(p); return sub(-np.median(z['lpips'], 1).astype(np.float64), z['idx'], COMX[tag][s])


# ---- Kamkari: LID at the calibrated threshold, PF-ODE likelihood of the main evaluation, dual-threshold AUROC
def lid_at(sv, thr): return (sv < thr).sum(1).astype(np.float64)


def calibrate(sv_train, target):
    lo, hi = -100.0, 100.0
    for _ in range(100):
        mid = 0.5 * (lo + hi); m = lid_at(sv_train, np.exp(-2 * mid)).mean()
        if m < target: hi = mid
        else: lo = mid
    return np.exp(-2 * 0.5 * (lo + hi))


def dual_auc_w(px, py, nx, ny, wp=None, wn=None):
    """area under the ROC chain of the two-threshold rule (x >= tx and y >= ty -> ID) over all threshold pairs, built exactly as
    kamkari_ana.py / the public roc_analysis.py: the (FP, TP) points of all pairs sorted by (FP, TP), a point kept when neither
    coordinate falls below the last kept point, trapezoids between consecutive kept points (plus (0,0) and (1,1)); with optional
    bootstrap multiplicities wp, wn.  Thresholds at every value of either set; the counts come from a 2-D cumulative histogram."""
    wp = np.ones(len(px)) if wp is None else wp; wn = np.ones(len(nx)) if wn is None else wn
    xs = np.unique(np.concatenate([px, nx])); ys = np.unique(np.concatenate([py, ny]))
    def grid(x, y, w):
        H = np.zeros((len(xs) + 1, len(ys) + 1)); ix = np.searchsorted(xs, x); iy = np.searchsorted(ys, y)
        np.add.at(H, (ix, iy), w)
        S = np.cumsum(np.cumsum(H[::-1, ::-1], 0), 1)[::-1, ::-1]                 # S[i, j] = weight with x >= xs[i] and y >= ys[j]
        return S[:len(xs), :len(ys)]
    fp = (grid(nx, ny, wn) / wn.sum()).ravel(); tp = (grid(px, py, wp) / wp.sum()).ravel()
    fp = np.round(fp, 10); tp = np.round(tp, 10)                                   # the reference sorts formatted 10-decimal keys
    o = np.lexsort((tp, fp)); fp = fp[o]; tp = tp[o]
    prev = np.concatenate([[0.0], np.maximum.accumulate(tp)[:-1]])                # max TP of the points before each one (chain start (0,0))
    keep = tp >= prev
    rx = np.concatenate([[0.0], fp[keep], [1.0]]); ry = np.concatenate([[0.0], tp[keep], [1.0]])
    return float(np.sum((rx[1:] - rx[:-1]) * (ry[1:] + ry[:-1]) / 2))


def kamkari_rows(tag):
    """(ll, lid) per set on the common images, or None if the tier is incomplete"""
    if tag not in LPCA_PUB: return None
    base = f'{ROOT}/runs/kamkari/{tag}_f2'; trp = f'{base}/{TRAIN[tag]}__k128.npz'
    if not os.path.exists(trp): return None
    sets = ['__id__'] + TIERS[tag]['ood']
    if not all(os.path.exists(f'{base}/{nm(tag, s)}.npz') for s in sets): return None
    thr = calibrate(np.load(trp)['sv'], LPCA_PUB[tag])
    out = {}
    for s in sets:
        z = np.load(f'{base}/{nm(tag, s)}.npz'); lid = sub(lid_at(z['sv'], thr), z['idx'], COMX[tag][s])
        out[s] = (nll(tag, s), lid)
    return out, float(thr)


t0 = time.time()
rng = np.random.default_rng(20261031)
OUT = {'split': SPLIT, 'B': B, 'pairs': {}, 'ids': {}, 'rows': {}, 'check_rotcls4': [], 'kamkari_thr': {}}
BO = {}; BR = {}
for tag, T in TIERS.items():
    oi = {sd: ours(tag, '__id__', sd) for sd in SEEDS}; Wa = counts(rng, len(oi[0]), B)
    gst = graham_stats(tag); gi = graham(tag, '__id__', gst); li = lmd(tag, '__id__')
    kam = kamkari_rows(tag)
    if kam: KR, thr = kam; OUT['kamkari_thr'][tag] = thr
    pr = {}
    for o in T['ood']:
        oo = {sd: ours(tag, o, sd) for sd in SEEDS}; Wb = counts(rng, len(oo[0]), B)
        ob = np.mean([boot_auroc(oi[sd], oo[sd], Wa, Wb) for sd in SEEDS], 0); opt = float(np.mean([100 * auroc(oi[sd], oo[sd]) for sd in SEEDS]))
        ref = REF['pairs'][tag][o]; assert abs(opt - ref['ours']) < 1e-9, (tag, o, opt, ref['ours'])
        if 'RotCls4' in ref['rows']:
            rb = boot_auroc(rotcls4(tag, '__id__'), rotcls4(tag, o), Wa, Wb); lo, hi = np.percentile(ob - rb, [2.5, 97.5])
            OUT['check_rotcls4'].append(max(abs(lo - ref['rows']['RotCls4']['ci'][0]), abs(hi - ref['rows']['RotCls4']['ci'][1])))
        rows = {}
        def add(r, a, b, bb):
            pt = 100 * auroc(a, b) if bb is not None else None
            d = ob - bb; lo, hi = np.percentile(d, [2.5, 97.5]); gap = opt - pt
            rows[r] = dict(auroc=float(pt), fpr95=float(100 * fpr95(a, b)), diff=float(gap), ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap)); BR[(tag, o, r)] = bb
        go = graham(tag, o, gst)
        if gi is not None and go is not None: add('Graham', gi, go, boot_auroc(gi, go, Wa, Wb))
        lo_ = lmd(tag, o)
        if li is not None and lo_ is not None: add('LMD', li, lo_, boot_auroc(li, lo_, Wa, Wb))
        if kam:
            (lli, lidi), (llo, lido) = KR['__id__'], KR[o]
            pt = 100 * dual_auc_w(lli, lidi, llo, lido)
            bb = np.array([100 * dual_auc_w(lli, lidi, llo, lido, Wa[r], Wb[r]) for r in range(B)])
            d = ob - bb; lo, hi = np.percentile(d, [2.5, 97.5]); gap = opt - pt
            rows['Kamkari'] = dict(auroc=float(pt), fpr95=None, diff=float(gap), ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap)); BR[(tag, o, 'Kamkari')] = bb
            print(f'   kamkari {tag} {o} {pt:.1f} ({gap:+.1f})  [{time.time() - t0:.0f}s]', flush=True)
        pr[o] = dict(ours=opt, rows=rows); BO[(tag, o)] = ob
    OUT['pairs'][tag] = pr
    nat = NAT[tag]; om = float(np.mean([pr[o]['ours'] for o in nat])); OUT['ids'][tag] = {'ours': om, 'rows': {}}
    for r in ('Graham', 'LMD', 'Kamkari'):
        if not all(r in pr[o]['rows'] for o in nat): continue
        rm = float(np.mean([pr[o]['rows'][r]['auroc'] for o in nat])); dn = np.mean([BO[(tag, o)] for o in nat], 0) - np.mean([BR[(tag, o, r)] for o in nat], 0)
        fm = [pr[o]['rows'][r]['fpr95'] for o in nat]; fm = float(np.mean(fm)) if all(v is not None for v in fm) else None
        lo, hi = np.percentile(dn, [2.5, 97.5]); OUT['ids'][tag]['rows'][r] = dict(mean=rm, fpr95=fm, diff=om - rm, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, om - rm))
    print(f"{ID_NAME[tag]:13s} Ours {om:5.1f}  " + '  '.join(f"{r} {v['mean']:5.1f} ({v['verdict']})" for r, v in OUT['ids'][tag]['rows'].items()) + f'  [{time.time() - t0:.0f}s]', flush=True)
assert max(OUT['check_rotcls4']) < 1e-9, max(OUT['check_rotcls4'])
for r in ('Graham', 'LMD', 'Kamkari'):
    vs = [p['rows'][r]['verdict'] for t in OUT['pairs'] for o, p in OUT['pairs'][t].items() if r in p['rows']]
    vn = [p['rows'][r]['verdict'] for t in OUT['pairs'] for o, p in OUT['pairs'][t].items() if r in p['rows'] and o in NAT[t]]
    if not vs: continue
    tiers = [t for t in TIERS if r in OUT['ids'][t]['rows']]
    OUT['rows'][r] = dict(wins=vs.count('win'), ties=vs.count('tie'), losses=vs.count('loss'), pairs=len(vs),
                          nat_wins=vn.count('win'), nat_ties=vn.count('tie'), nat_losses=vn.count('loss'), nat_pairs=len(vn),
                          tiers=tiers, mean_nat=float(np.mean([OUT['ids'][t]['rows'][r]['mean'] for t in tiers])))
    print(r, OUT['rows'][r], flush=True)
json.dump(OUT, open(OUTPATH, 'w'), indent=1)
print('saved', OUTPATH, f'[{time.time() - t0:.0f}s]', flush=True)
