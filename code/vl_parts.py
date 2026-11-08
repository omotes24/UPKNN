# vl_parts.py -- what the Vertical-Limit gain is made of (development split by default; never used to choose anything):
#   rows built from the per-image terms of the frozen scores (data_kx/export: Upright score, pose; data_vl/export: deep0, deepk),
#   noise seeds 0-2, the bootstrap protocol of ours_vs_all.py (2000, rng 20261031; W/T/L against Upright kNN):
#     VL        = U + a0 (deep0 + pose) + w* (deepk + pose)          (the frozen score, as exported)
#     VL-sigma0 = U + a0 (deep0 + pose)                              (no second network evaluation)
#     pose-only = U + (a0 + w*) pose                                  (the pose term up-weighted as in VL, no deep-level kNN)
#     deep-only = U + a0 deep0 + w* deepk                             (the deep-level kNN terms, the pose term not repeated)
#   usage: SPLIT=fresh python vl_parts.py
import os, sys, json
import numpy as np
SPLIT = os.environ.get('SPLIT', 'fresh'); assert SPLIT in ('fresh', 'final2')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../ana')
os.environ['SPLIT'] = SPLIT
from ana_common import *
from boot import counts, boot_auroc
B = 2000; SEEDS = (0, 1, 2)
COMX = json.load(open(f'{SCR}/final/eu_stage/push/fresh_idx/fresh_idx.json')) if SPLIT == 'fresh' else json.load(open(f'{SCR}/final/eu_stage/push_final2/final2_idx/final2_idx.json'))
EXP_U = f'{SCR}/final/data_kx/export'; EXP_V = f'{SCR}/final/data_vl/export'
P = json.load(open(f'{SCR}/final/data_vl/json/PROTOCOL_VL.json'))


def verdict(lo, hi, gap):
    if abs(gap) < 0.05: return 'tie'
    return 'win' if lo > 0 else ('loss' if hi < 0 else 'tie')


def get(exp, tag, s, seed, key):
    Z = np.load(f'{exp}/{tag}_s{seed}_{SPLIT}.npz'); n = f'{SPLIT}-{setname(tag, s)}'
    return sub(Z[f'{key}__{n}'].astype(np.float64), Z[f'idx__{n}'], COMX[tag][s])


def rows(tag, s, seed):
    a0, w = P[tag]['a0'], P[tag]['wstar']
    U = get(EXP_U, tag, s, seed, 'score'); q = get(EXP_U, tag, s, seed, 'pose')
    d0 = get(EXP_V, tag, s, seed, 'deep0'); dk = get(EXP_V, tag, s, seed, 'deepk'); V = get(EXP_V, tag, s, seed, 'score')
    assert np.allclose(V, U + a0 * (d0 + q) + w * (dk + q)), (tag, s, seed)
    return {'Upright': U, 'VL': V, 'VL-sigma0': U + a0 * (d0 + q), 'pose-only': U + (a0 + w) * q, 'deep-only': U + a0 * d0 + w * dk}


rng = np.random.default_rng(20261031)
NAMES = ['Upright', 'VL', 'VL-sigma0', 'pose-only', 'deep-only']
OUT = {'split': SPLIT, 'B': B, 'pairs': {}, 'ids': {}, 'wtl': {}}
allv = {r: [] for r in NAMES[1:]}
for tag, T in TIERS.items():
    ri = {sd: rows(tag, '__id__', sd) for sd in SEEDS}; Wa = counts(rng, len(ri[0]['Upright']), B); pr = {}; bo = {}
    for o in T['ood']:
        ro = {sd: rows(tag, o, sd) for sd in SEEDS}; Wb = counts(rng, len(ro[0]['Upright']), B); pr[o] = {}
        for r in NAMES:
            pts = [100 * auroc(ri[sd][r], ro[sd][r]) for sd in SEEDS]
            bo[(o, r)] = np.mean([boot_auroc(ri[sd][r], ro[sd][r], Wa, Wb) for sd in SEEDS], 0)
            pr[o][r] = {'auroc': float(np.mean(pts))}
            if r != 'Upright':
                d = bo[(o, r)] - bo[(o, 'Upright')]; lo, hi = np.percentile(d, [2.5, 97.5]); gap = pr[o][r]['auroc'] - pr[o]['Upright']['auroc']
                pr[o][r].update(diff=float(gap), ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap)); allv[r].append(pr[o][r]['verdict'])
    OUT['pairs'][tag] = pr
    OUT['ids'][tag] = {r: float(np.mean([pr[o][r]['auroc'] for o in NAT[tag]])) for r in NAMES}
    print(f"{ID_NAME[tag]:13s} " + '  '.join(f"{r} {OUT['ids'][tag][r]:6.2f}" for r in NAMES), flush=True)
OUT['mean'] = {r: float(np.mean([OUT['ids'][t][r] for t in TIERS])) for r in NAMES}
OUT['wtl'] = {r: dict(wins=v.count('win'), ties=v.count('tie'), losses=v.count('loss'), pairs=len(v)) for r, v in allv.items()}
print('6-ID natural mean:', {r: round(v, 2) for r, v in OUT['mean'].items()})
print('W/T/L vs Upright:', OUT['wtl'])
near = [('edmc10', 'cifar100_test'), ('edmc100', 'cifar10_test'), ('in64', 'in64_ssb_hard'), ('in64', 'in64_ninco')]
for t, o in near: print('near', t, o, {r: round(OUT['pairs'][t][o][r]['auroc'], 2) for r in NAMES})
json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'vl_parts_{SPLIT}.json'), 'w'), indent=1)
