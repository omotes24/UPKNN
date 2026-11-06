# e7_ana.py -- the resampling control of PROTOCOL_HIRES2.md (tier in256d: ADM 256, every image through 112 px by one BOX reduction and one
#   bicubic upsampling), evaluated exactly as hires_ana.py evaluates in256c, and compared with in256c on the images both tiers share.
#   Rows: Ours (3 seeds), kNN term, orientation term, kNN (basic) (3 seeds); CFS diag / kNN (3 seeds).  Bootstrap 2000, rng 20261031, the
#   verdict rule of the paper.   usage: python e7_ana.py <in256d results dir> [<in256c results dir>]   -> e7_ana.json
import os, sys, json
import numpy as np
SCR = os.environ.get('UPKNN_ANALYSIS', os.path.expanduser('~/upknn/analysis'))
HERE = os.path.dirname(os.path.abspath(__file__))
RES = {'in256d': sys.argv[1] if len(sys.argv) > 1 else f'{SCR}/final/e7/res', 'in256c': sys.argv[2] if len(sys.argv) > 2 else f'{SCR}/final/hires/res'}
sys.path.insert(0, f'{SCR}/eu_build/Euphonium/analysis'); sys.path.insert(0, f'{SCR}/final/ana')
from common import sub, auroc, fpr95
from boot import counts, boot_auroc
NEAR = ['ssb_hard', 'ninco']; FAR = ['inaturalist', 'textures', 'openimage_o']; OOD = NEAR + FAR; VF = 'imagenet_test__vflip'
CS = ['imagenet_v2', 'imagenet_c', 'imagenet_r']; ID = 'imagenet_test'; ALLS = [ID, VF] + OOD + CS
NOPENOOD = {ID: 45000, 'imagenet_v2': 10000, 'imagenet_c': 10000, 'imagenet_r': 30000}
PFX = {'in256c': 'q256-', 'in256d': 'q256d-'}
SEEDS = (0, 1, 2)


def rows_for(tag):
    R = {}; root = RES[tag]
    for sd in SEEDS:
        fx = f'{root}/runs/kx/export/{tag}_s{sd}_{PFX[tag].replace("-", "")}.npz'
        if not os.path.exists(fx): continue
        U = np.load(fx); K = np.load(f'{root}/runs/hires/knn_{tag}_s{sd}.npz')
        for s in ALLS:
            n = PFX[tag] + s
            for r, z, key in (('Ours', U, 'score'), ('kNN term', U, 'content'), ('Orientation term', U, 'pose'), ('kNN basic', K, 'ct')):
                R.setdefault(r, {}).setdefault(sd, {})[s] = (np.asarray(z[f'{key}__{n}'], np.float64), np.asarray(z[f'idx__{n}']))
        cd = f'{root}/runs/cfs_l7/s{sd}/{tag}'
        if os.path.isdir(cd) and all(os.path.exists(f'{cd}/{PFX[tag]}{s}__scores.npz') for s in ALLS):
            for s in ALLS:
                C = np.load(f'{cd}/{PFX[tag]}{s}__scores.npz'); ix = np.asarray(C['idx'])
                R.setdefault('CFS', {}).setdefault(sd, {})[s] = (C['diag'].astype(np.float64), ix)
                R.setdefault('CFS kNN', {}).setdefault(sd, {})[s] = (C['knn'].astype(np.float64), ix)
    return R


def wauroc(a, wa, b):
    bs = np.sort(b); f = (np.searchsorted(bs, a, 'left') + 0.5 * (np.searchsorted(bs, a, 'right') - np.searchsorted(bs, a, 'left'))) / len(b)
    return 100 * float(np.sum(wa * f) / np.sum(wa))


def metrics(get):
    a = get(ID); out = {}
    for s in OOD + [VF] + CS:
        b = get(s); out[s] = dict(auroc=100 * auroc(a, b), fpr95=100 * fpr95(a, b))
    for g, L in (('near', NEAR), ('far', FAR)):
        out[g] = float(np.mean([out[s]['auroc'] for s in L])); out[g + '_fpr95'] = float(np.mean([out[s]['fpr95'] for s in L]))
    pos = np.concatenate([get(s) for s in [ID] + CS]); w = np.concatenate([np.full(len(get(s)), NOPENOOD[s] / len(get(s))) for s in [ID] + CS])
    for s in OOD: out['FS-' + s] = dict(auroc=wauroc(pos, w, get(s)))
    for g, L in (('near', NEAR), ('far', FAR)): out['FS-' + g] = float(np.mean([out['FS-' + s]['auroc'] for s in L]))
    return out


def avg(dicts):
    if len(dicts) == 1: return dicts[0]
    o = {}
    for k, v in dicts[0].items():
        o[k] = {kk: float(np.mean([d[k][kk] for d in dicts])) for kk in v} if isinstance(v, dict) else float(np.mean([d[k] for d in dicts]))
    return o


def verdict(lo, hi, gap):
    if abs(gap) < 0.05: return 'tie'
    return 'win' if lo > 0 else ('loss' if hi < 0 else 'tie')


OUT = {}; ALLROWS = {}
for tag in ('in256d', 'in256c'):
    R = rows_for(tag)
    if 'Ours' not in R: print(tag, 'no results yet'); continue
    ALLROWS[tag] = R
    order = [r for r in ['Ours', 'kNN term', 'Orientation term', 'kNN basic', 'CFS', 'CFS kNN'] if r in R]
    COM = {s: sorted(set.intersection(*[set(R[r][sd][s][1].tolist()) for r in order for sd in R[r]])) for s in ALLS}
    T = {'seeds': {r: sorted(R[r]) for r in order}, 'n_common': {s: len(v) for s, v in COM.items()}, 'common': {}, 'boot': {}}
    for r in order: T['common'][r] = avg([metrics(lambda s, sd=sd: sub(*R[r][sd][s], COM[s])) for sd in sorted(R[r])])
    rng = np.random.default_rng(20261031); B = 2000
    Wa = counts(rng, len(COM[ID]), B); Wb = {s: counts(rng, len(COM[s]), B) for s in OOD + [VF]}
    def boots(r): return {s: np.mean([boot_auroc(sub(*R[r][sd][ID], COM[ID]), sub(*R[r][sd][s], COM[s]), Wa, Wb[s]) for sd in sorted(R[r])], 0) for s in OOD + [VF]}
    bo = boots('Ours')
    for r in order[1:]:
        br = boots(r); res = {}
        for s in OOD + [VF]:
            d = bo[s] - br[s]; lo, hi = np.percentile(d, [2.5, 97.5]); gap = T['common']['Ours'][s]['auroc'] - T['common'][r][s]['auroc']
            res[s] = dict(diff=gap, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap))
        for g, L in (('near', NEAR), ('far', FAR)):
            d = np.mean([bo[s] for s in L], 0) - np.mean([br[s] for s in L], 0); lo, hi = np.percentile(d, [2.5, 97.5])
            gap = T['common']['Ours'][g] - T['common'][r][g]; res[g] = dict(diff=gap, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap))
        vs = [res[s]['verdict'] for s in OOD]; res['wtl'] = [vs.count('win'), vs.count('tie'), vs.count('loss')]
        T['boot'][r] = res
    T['temp'] = {}
    for sd in SEEDS:
        p = f'{RES[tag]}/runs/kx/rothead/s{sd}/{tag}_k0_gqR/temp.json'
        if os.path.exists(p): T['temp'][sd] = json.load(open(p))
    OUT[tag] = T
    print(f'== {tag}: common images', T['n_common'])
    print('   SSB NINCO | near | iNat Tex OIO | far | flip | V2 C R | FS near far | W/T/L')
    for r in order:
        M = T['common'][r]; ex = '' if r == 'Ours' else f"  {T['boot'][r]['wtl']} near {T['boot'][r]['near']['verdict']} far {T['boot'][r]['far']['verdict']}"
        print(f"{r:17s}" + ' '.join(f"{M[s]['auroc']:5.1f}" for s in NEAR) + f" | {M['near']:5.1f} | " + ' '.join(f"{M[s]['auroc']:5.1f}" for s in FAR) +
              f" | {M['far']:5.1f} | {M[VF]['auroc']:5.1f} | " + ' '.join(f"{M[s]['auroc']:5.1f}" for s in CS) + f" | {M['FS-near']:5.1f} {M['FS-far']:5.1f}{ex}")
# the two tiers on the images both have: every row that both tiers have, with a paired bootstrap of in256d - in256c per set
if len(ALLROWS) == 2:
    rows = [r for r in ['Ours', 'kNN term', 'Orientation term', 'kNN basic', 'CFS', 'CFS kNN'] if r in ALLROWS['in256c'] and r in ALLROWS['in256d']]
    COMX = {s: sorted(set.intersection(*[set(ALLROWS[t][r][sd][s][1].tolist()) for t in ALLROWS for r in rows for sd in ALLROWS[t][r]])) for s in ALLS}
    X = {'n_common': {s: len(v) for s, v in COMX.items()}, 'rows': {}, 'boot': {}}
    rng = np.random.default_rng(20261031); B = 2000
    Wa = counts(rng, len(COMX[ID]), B); Wb = {s: counts(rng, len(COMX[s]), B) for s in OOD + [VF]}
    for r in rows:
        X['rows'][r] = {t: avg([metrics(lambda s, sd=sd, t=t: sub(*ALLROWS[t][r][sd][s], COMX[s])) for sd in sorted(ALLROWS[t][r])]) for t in ALLROWS}
        bt = {t: {s: np.mean([boot_auroc(sub(*ALLROWS[t][r][sd][ID], COMX[ID]), sub(*ALLROWS[t][r][sd][s], COMX[s]), Wa, Wb[s]) for sd in sorted(ALLROWS[t][r])], 0)
                  for s in OOD + [VF]} for t in ALLROWS}
        res = {}
        for s in OOD + [VF]:
            d = bt['in256d'][s] - bt['in256c'][s]; lo, hi = np.percentile(d, [2.5, 97.5]); gap = X['rows'][r]['in256d'][s]['auroc'] - X['rows'][r]['in256c'][s]['auroc']
            res[s] = dict(diff=gap, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap))
        for g, L in (('near', NEAR), ('far', FAR)):
            d = np.mean([bt['in256d'][s] for s in L], 0) - np.mean([bt['in256c'][s] for s in L], 0); lo, hi = np.percentile(d, [2.5, 97.5])
            gap = X['rows'][r]['in256d'][g] - X['rows'][r]['in256c'][g]; res[g] = dict(diff=gap, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap))
        X['boot'][r] = res
    OUT['cross'] = X
    print('== in256d vs in256c on the common images (in256d - in256c; verdict of the paired bootstrap)')
    for r in rows:
        print(f"{r:17s}" + ' '.join(f"{s[:5]:>6s} {X['rows'][r]['in256d'][s]['auroc']:5.1f}/{X['rows'][r]['in256c'][s]['auroc']:5.1f} ({X['boot'][r][s]['diff']:+.1f} {X['boot'][r][s]['verdict']})" for s in OOD + [VF]))
json.dump(OUT, open(f'{HERE}/e7_ana.json', 'w'), indent=1)
print('written', f'{HERE}/e7_ana.json')
