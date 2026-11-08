# vl_vs_uknn.py -- Vertical Limit (runs/vl/export via data_vl/export, noise seeds 0-2) against Upright kNN (kx_export.py, the same seeds and
#   images) on one split, with the bootstrap protocol of ours_vs_all.py (2000 replicates, rng 20261031, ID images resampled once per ID,
#   OOD per set, in TIERS order; per pair CI of AUROC(VL) - AUROC(Upright); win/loss if the CI excludes 0 and |gap| >= 0.05).
#   usage: SPLIT=fresh python vl_vs_uknn.py      (SPLIT=final2 only after the protocol is frozen)
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


def verdict(lo, hi, gap):
    if abs(gap) < 0.05: return 'tie'
    return 'win' if lo > 0 else ('loss' if hi < 0 else 'tie')


def get(exp, tag, s, seed, key='score'):
    Z = np.load(f'{exp}/{tag}_s{seed}_{SPLIT}.npz'); n = f'{SPLIT}-{setname(tag, s)}'
    return sub(Z[f'{key}__{n}'].astype(np.float64), Z[f'idx__{n}'], COMX[tag][s])


rng = np.random.default_rng(20261031)
OUT = {'split': SPLIT, 'B': B, 'pairs': {}, 'ids': {}}
ONLY = os.environ.get('TAGS', '').split(',') if os.environ.get('TAGS') else list(TIERS)
for tag, T in TIERS.items():
    if tag not in ONLY: continue
    vi = {sd: get(EXP_V, tag, '__id__', sd) for sd in SEEDS}; ui = {sd: get(EXP_U, tag, '__id__', sd) for sd in SEEDS}
    Wa = counts(rng, len(vi[0]), B); pr, bv, bu = {}, {}, {}
    for o in T['ood']:
        vo = {sd: get(EXP_V, tag, o, sd) for sd in SEEDS}; uo = {sd: get(EXP_U, tag, o, sd) for sd in SEEDS}; Wb = counts(rng, len(vo[0]), B)
        pv = [100 * auroc(vi[sd], vo[sd]) for sd in SEEDS]; pu = [100 * auroc(ui[sd], uo[sd]) for sd in SEEDS]
        bv[o] = np.mean([boot_auroc(vi[sd], vo[sd], Wa, Wb) for sd in SEEDS], 0); bu[o] = np.mean([boot_auroc(ui[sd], uo[sd], Wa, Wb) for sd in SEEDS], 0)
        lo, hi = np.percentile(bv[o] - bu[o], [2.5, 97.5]); gap = float(np.mean(pv) - np.mean(pu))
        pr[o] = dict(vl=float(np.mean(pv)), vl_seeds=pv, upright=float(np.mean(pu)), upright_seeds=pu, diff=gap, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap))
    OUT['pairs'][tag] = pr
    nat = NAT[tag]; vm = float(np.mean([pr[o]['vl'] for o in nat])); um = float(np.mean([pr[o]['upright'] for o in nat]))
    dn = np.mean([bv[o] for o in nat], 0) - np.mean([bu[o] for o in nat], 0); lo, hi = np.percentile(dn, [2.5, 97.5])
    OUT['ids'][tag] = dict(vl=vm, upright=um, diff=vm - um, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, vm - um))
    vs = [p['verdict'] for p in pr.values()]
    print(f"{ID_NAME[tag]:13s} VL {vm:6.2f}  Upright {um:6.2f}  diff {vm - um:+.2f}  W/T/L {vs.count('win')}/{vs.count('tie')}/{vs.count('loss')} | "
          + ' '.join(f"{o[:10]}:{p['vl']:.1f}/{p['upright']:.1f}{p['verdict'][0].upper()}" for o, p in pr.items()), flush=True)
allv = [p['verdict'] for t in OUT['pairs'] for p in OUT['pairs'][t].values()]
OUT['wtl'] = dict(wins=allv.count('win'), ties=allv.count('tie'), losses=allv.count('loss'), pairs=len(allv))
OUT['mean_vl'] = float(np.mean([OUT['ids'][t]['vl'] for t in OUT['ids']])); OUT['mean_upright'] = float(np.mean([OUT['ids'][t]['upright'] for t in OUT['ids']]))
print('6-ID natural mean: VL', round(OUT['mean_vl'], 2), 'Upright', round(OUT['mean_upright'], 2), '| W/T/L', OUT['wtl'])
json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'vl_vs_uknn_{SPLIT}' + ('' if len(OUT['ids']) == len(TIERS) else '_part') + '.json'), 'w'), indent=1)
