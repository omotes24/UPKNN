# vl_ctrl_ana.py -- the two controls of the Vertical-Limit section on the development split (noise seeds 0-2, the bootstrap protocol of
#   ours_vs_all.py: 2000 replicates, rng 20261031, W/T/L when the 95% interval excludes 0 and |gap| >= 0.05), from
#   data_vl/ctrl/<tag>_s<seed>_fresh.npz (code/vl_ctrl.py on the GPU server: cA = deep level at sigma_1 with R*, cBA = conditional kNN, cB = the
#   Upright content recomputed, cM = channel-mean expert) and the frozen exports (data_kx/export: Upright score, content, pose; data_vl/export).
#   rows:  Upright | VL | VL-sigma1 = Upright + a0 (cA + pose)  (VL without the second network evaluation; the product of experts)
#          chain = cA + cBA + pose if a0 = 1, else Upright  (control 1, conditional kNN: the same two features as VL-sigma1 as one
#                  density of (A, B) by the chain rule, the pose term once)
#          pose-only = Upright + (a0 + w*) pose  (the pose term up-weighted as in VL, no deep-level kNN)
#          VL+mean = VL + (cM + pose)  (control 2: a third expert, the channel means of the deep level)
#   usage: python vl_ctrl_ana.py      -> vl_ctrl_fresh.json
import os, sys, json
import numpy as np
SPLIT = 'fresh'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../ana')
os.environ['SPLIT'] = SPLIT
from ana_common import *
from boot import counts, boot_auroc
B = 2000; SEEDS = (0, 1, 2)
COMX = json.load(open(f'{SCR}/final/eu_stage/push/fresh_idx/fresh_idx.json'))
EXP_U = f'{SCR}/final/data_kx/export'; EXP_V = f'{SCR}/final/data_vl/export'; CT = f'{SCR}/final/data_vl/ctrl'


def verdict(lo, hi, gap):
    if abs(gap) < 0.05: return 'tie'
    return 'win' if lo > 0 else ('loss' if hi < 0 else 'tie')


def get(path, tag, s, seed, key):
    Z = np.load(f'{path}/{tag}_s{seed}_{SPLIT}.npz'); n = f'{SPLIT}-{setname(tag, s)}'
    return sub(Z[f'{key}__{n}'].astype(np.float64), Z[f'idx__{n}'], COMX[tag][s])


P = json.load(open(f'{SCR}/final/data_vl/json/PROTOCOL_VL.json'))
CHECK, CHECKA = {}, {}
def rows(tag, s, seed):
    a0, w = P[tag]['a0'], P[tag]['wstar']
    U = get(EXP_U, tag, s, seed, 'score'); q = get(EXP_U, tag, s, seed, 'pose'); V = get(EXP_V, tag, s, seed, 'score')
    cA, cBA, cB, cM = (get(CT, tag, s, seed, k) for k in ('cA', 'cBA', 'cB', 'cM'))
    c_exp = get(EXP_U, tag, s, seed, 'content'); CHECK.setdefault(tag, []).append(float(np.max(np.abs(cB - c_exp))))   # recomputed content = exported
    if a0: CHECKA.setdefault(tag, []).append(float(np.max(np.abs(cA - get(EXP_V, tag, s, seed, 'deep0')))))          # recomputed deep term = exported
    return {'Upright': U, 'VL': V, 'VL-sigma1': U + a0 * (cA + q), 'chain': cA + cBA + q if a0 else U, 'pose-only': U + (a0 + w) * q, 'VL+mean': V + cM + q}


rng = np.random.default_rng(20261031)
NAMES = ['Upright', 'VL', 'VL-sigma1', 'chain', 'pose-only', 'VL+mean']
OUT = {'split': SPLIT, 'B': B, 'pairs': {}, 'ids': {}, 'wtl': {}, 'wtl_vs_vl': {}, 'choices': {}}
allv = {r: [] for r in NAMES[1:]}; allvv = {r: [] for r in NAMES[2:]}
for tag, T in TIERS.items():
    OUT['choices'][tag] = json.load(open(f'{CT}/{tag}_choice.json'))
    ri = {sd: rows(tag, '__id__', sd) for sd in SEEDS}; Wa = counts(rng, len(ri[0]['Upright']), B); pr = {}; bo = {}
    for o in T['ood']:
        ro = {sd: rows(tag, o, sd) for sd in SEEDS}; Wb = counts(rng, len(ro[0]['Upright']), B); pr[o] = {}
        for r in NAMES:
            pts = [100 * auroc(ri[sd][r], ro[sd][r]) for sd in SEEDS]
            bo[(o, r)] = np.mean([boot_auroc(ri[sd][r], ro[sd][r], Wa, Wb) for sd in SEEDS], 0); pr[o][r] = {'auroc': float(np.mean(pts))}
            for ref, acc in (('Upright', allv), ('VL', allvv)):
                if r in (ref, 'Upright') or (ref == 'VL' and r == 'VL'): continue
                d = bo[(o, r)] - bo[(o, ref)]; lo, hi = np.percentile(d, [2.5, 97.5]); gap = pr[o][r]['auroc'] - pr[o][ref]['auroc']
                pr[o][r][f'vs_{ref}'] = dict(diff=float(gap), ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap)); acc[r].append(pr[o][r][f'vs_{ref}']['verdict'])
    OUT['pairs'][tag] = pr
    OUT['ids'][tag] = {r: float(np.mean([pr[o][r]['auroc'] for o in NAT[tag]])) for r in NAMES}
    print(f"{ID_NAME[tag]:13s} " + '  '.join(f"{r} {OUT['ids'][tag][r]:6.2f}" for r in NAMES) + f"   |cB-content| max {max(CHECK[tag]):.2e} |cA-deep0| max {max(CHECKA.get(tag, [0])):.2e}", flush=True)
OUT['mean'] = {r: float(np.mean([OUT['ids'][t][r] for t in TIERS])) for r in NAMES}
OUT['wtl'] = {r: dict(wins=v.count('win'), ties=v.count('tie'), losses=v.count('loss'), pairs=len(v)) for r, v in allv.items()}
OUT['wtl_vs_vl'] = {r: dict(wins=v.count('win'), ties=v.count('tie'), losses=v.count('loss'), pairs=len(v)) for r, v in allvv.items()}
OUT['check_content_maxabs'] = {t: max(v) for t, v in CHECK.items()}; OUT['check_deep0_maxabs'] = {t: max(v) for t, v in CHECKA.items()}
print('6-ID natural mean:', {r: round(v, 2) for r, v in OUT['mean'].items()})
print('W/T/L vs Upright:', OUT['wtl']); print('W/T/L vs VL:', OUT['wtl_vs_vl'])
for t, o in [('edmc10', 'cifar100_test'), ('edmc100', 'cifar10_test'), ('in64', 'in64_ssb_hard'), ('in64', 'in64_ninco')]:
    print('near', t, o, {r: round(OUT['pairs'][t][o][r]['auroc'], 2) for r in NAMES})
json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vl_ctrl_fresh.json'), 'w'), indent=1)
