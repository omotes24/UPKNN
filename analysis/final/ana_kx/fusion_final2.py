# fusion_final2.py -- E2 of the IJCAI revision: is Upright kNN just "kNN + a rotation classifier"?  Naive fusions of a kNN score and an
#   orientation score against the principled combination of Ours (score = content + pose = -d log r_R + log q_T(up|x)), on final2,
#   same images and bootstrap protocol as ours_vs_all.py (2000 replicates, rng 20261031, win/loss if the CI of AUROC(Ours) - AUROC(row)
#   excludes 0 and |gap| >= 0.05).  Normalisation statistics of every fusion come from the ID images of the fresh split (unlabelled ID
#   images disjoint from final2); nothing is fitted on OOD data.
#   Rows:  kNN term (content), Orientation term (pose), RotCls4 (ResNet-18 rotation classifier, image space, 1 seed),
#          kNN+RotCls4 (z-sum), kNN+RotCls4 (Fisher), kNN+Orient. (z-sum), kNN+Orient. (Fisher), Ours with d = 1 (content / d + pose).
#   usage: python fusion_final2.py   -> fusion_final2.json
import os, sys, json
import numpy as np
SPLIT = 'final2'; os.environ['SPLIT'] = SPLIT
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE + '/../ana'); sys.path.insert(0, HERE + '/../ana_ok')
from ana_common import *
from boot import counts, boot_auroc
B = 2000; SEEDS = (0, 1, 2)
COMX = json.load(open(f'{SCR}/final/eu_stage/push_final2/final2_idx/final2_idx.json'))
EXP = f'{SCR}/final/data_kx/export'
def nm(tag, s): return setname(tag, s)
def exp(tag, seed, split): return np.load(f'{EXP}/{tag}_s{seed}_{split}.npz')
def kx(tag, s, seed, key, split=SPLIT, common=True):
    Z = exp(tag, seed, split); n = f'{split}-{nm(tag, s)}'; v = Z[f'{key}__{n}'].astype(np.float64)
    return sub(v, Z[f'idx__{n}'], COMX[tag][s]) if common else v
def rc(tag, s, split=SPLIT, common=True):
    z = np.load(f'{DATA}/runs/rotcls/{tag}/scores_{split}/{nm(tag, s)}.npz'); v = z['rot4'].astype(np.float64)
    return sub(v, z['idx'], COMX[tag][s]) if common else v
class Norm:
    """ID-only normalisation from the fresh-split ID scores: z-score and the empirical ID CDF (for Fisher's method)"""
    def __init__(s, ref): s.ref = np.sort(ref); s.mu, s.sd = ref.mean(), ref.std() + 1e-12
    def z(s, v): return (v - s.mu) / s.sd
    def logF(s, v): return np.log((np.searchsorted(s.ref, v, 'right') + 1.0) / (len(s.ref) + 1.0))
def rows_for(tag, seed):
    """{row: function(set) -> ID-high scores on the common final2 images} for one ID and noise seed"""
    d = float(exp(tag, seed, SPLIT)['d'])
    Nc = Norm(kx(tag, '__id__', seed, 'content', 'fresh', False)); Np = Norm(kx(tag, '__id__', seed, 'pose', 'fresh', False))
    Nr = Norm(rc(tag, '__id__', 'fresh', False))
    c = lambda s: kx(tag, s, seed, 'content'); p = lambda s: kx(tag, s, seed, 'pose'); r = lambda s: rc(tag, s)
    return {'Ours': lambda s: kx(tag, s, seed, 'score'),
            'kNN term': c, 'Orientation term': p, 'RotCls4': r,
            'kNN+RotCls4 (z-sum)': lambda s: Nc.z(c(s)) + Nr.z(r(s)),
            'kNN+RotCls4 (Fisher)': lambda s: Nc.logF(c(s)) + Nr.logF(r(s)),
            'kNN+Orient. (z-sum)': lambda s: Nc.z(c(s)) + Np.z(p(s)),
            'kNN+Orient. (Fisher)': lambda s: Nc.logF(c(s)) + Np.logF(p(s)),
            'Ours, d = 1': lambda s: c(s) / d + p(s)}
def verdict(lo, hi, gap):
    if abs(gap) < 0.05: return 'tie'
    return 'win' if lo > 0 else ('loss' if hi < 0 else 'tie')
rng = np.random.default_rng(20261031)
OUT = {'split': SPLIT, 'B': B, 'ids': {}, 'pairs': {}, 'vflip': {}}
NAMES = None
for tag, T in TIERS.items():
    R = {sd: rows_for(tag, sd) for sd in SEEDS}; NAMES = list(R[0])
    ri = {r: {sd: R[sd][r]('__id__') for sd in SEEDS} for r in NAMES}
    Wa = counts(rng, len(ri['Ours'][0]), B)
    pr, bo = {}, {}
    for o in T['ood']:
        ro = {r: {sd: R[sd][r](o) for sd in SEEDS} for r in NAMES}; Wb = counts(rng, len(ro['Ours'][0]), B)
        pts = {r: [100 * auroc(ri[r][sd], ro[r][sd]) for sd in SEEDS] for r in NAMES}
        bb = {r: np.mean([boot_auroc(ri[r][sd], ro[r][sd], Wa, Wb) for sd in SEEDS], 0) for r in NAMES}
        pr[o] = {r: dict(auroc=float(np.mean(pts[r])), seeds=pts[r]) for r in NAMES}
        for r in NAMES[1:]:
            dd = bb['Ours'] - bb[r]; lo, hi = np.percentile(dd, [2.5, 97.5]); gap = pr[o]['Ours']['auroc'] - pr[o][r]['auroc']
            pr[o][r].update(diff=gap, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap))
        bo[o] = bb
    OUT['pairs'][tag] = pr
    nat = NAT[tag]
    OUT['ids'][tag] = {r: float(np.mean([pr[o][r]['auroc'] for o in nat])) for r in NAMES}
    vf = [o for o in T['ood'] if 'vflip' in o]
    OUT['vflip'][tag] = {r: float(np.mean([pr[o][r]['auroc'] for o in vf])) for r in NAMES} if vf else None
    print(f"{ID_NAME[tag]:13s} " + ' '.join(f"{r[:14]:>14s} {OUT['ids'][tag][r]:5.1f}" for r in NAMES), flush=True)
OUT['mean'] = {r: float(np.mean([OUT['ids'][t][r] for t in TIERS])) for r in NAMES}
OUT['vflip_mean'] = {r: float(np.mean([OUT['vflip'][t][r] for t in TIERS if OUT['vflip'][t]])) for r in NAMES}
allp = [(t, o) for t in OUT['pairs'] for o in OUT['pairs'][t] if o in NAT[t]]
OUT['wtl_nat'] = {r: {v: [OUT['pairs'][t][o][r]['verdict'] for t, o in allp].count(v) for v in ('win', 'tie', 'loss')} for r in NAMES[1:]}
allv = [(t, o) for t in OUT['pairs'] for o in OUT['pairs'][t]]
OUT['wtl_all'] = {r: {v: [OUT['pairs'][t][o][r]['verdict'] for t, o in allv].count(v) for v in ('win', 'tie', 'loss')} for r in NAMES[1:]}
print('mean over IDs (natural OOD):', {r: round(v, 1) for r, v in OUT['mean'].items()})
print('vflip mean:', {r: round(v, 1) for r, v in OUT['vflip_mean'].items()})
print('Ours vs row, W/T/L natural pairs:', OUT['wtl_nat']); print('all pairs:', OUT['wtl_all'])
json.dump(OUT, open(os.path.join(HERE, 'fusion_final2.json'), 'w'), indent=1)
