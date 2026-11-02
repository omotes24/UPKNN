# in256_ana.py -- the ImageNet-256 tier (PROTOCOL_IN256.md): Upright kNN with the ADM 256x256 unconditional model on the OpenOOD v1.5
#   ImageNet-1K sets (ID = ImageNet val, near = SSB-hard, NINCO, far = iNaturalist, Textures, OpenImage-O; vertically flipped ID images;
#   covariate-shifted ID = ImageNet-V2, -C, -R), frozen ImageNet-64 configuration, noise seeds 0-2.
#   Rows: Ours, its two terms, kNN (basic), CFS (diag / kNN heads) from runs of this tier; MSMA*, DiffPath, GEPC from the stored statistics
#   of the 256 tier (Euphonium analysis/baselines_eval.py, fits on the ID calibration images, unchanged; 500 images per set).
#   Comparison table on the images every row has (common images); bootstrap 2000 replicates, rng 20261031, verdict as everywhere.
#   Full spectrum: ID + csID positive, each image weighted by (OpenOOD set size) / (number of images used) of its set.
#   output: in256_ana.json (+ printed tables)
import os, sys, json
import numpy as np
SCR = os.environ.get('UPKNN_ANALYSIS', os.path.expanduser('~/upknn/analysis'))
HERE = os.path.dirname(os.path.abspath(__file__)); RES = f'{SCR}/final/in256/res'
os.environ['EUPHONIUM_INPUTS'] = f'{SCR}/eu_build/inputs_full'
sys.path.insert(0, f'{SCR}/eu_build/Euphonium/analysis'); sys.path.insert(0, f'{SCR}/final/ana')
import baselines_eval as BE
from common import sub, auroc, fpr95
from boot import counts, boot_auroc
NEAR = ['ssb_hard', 'ninco']; FAR = ['inaturalist', 'textures', 'openimage_o']; OOD = NEAR + FAR; VF = 'imagenet_test__vflip'
CS = ['imagenet_v2', 'imagenet_c', 'imagenet_r']; ID = 'imagenet_test'
NOPENOOD = {ID: 45000, 'imagenet_v2': 10000, 'imagenet_c': 10000, 'imagenet_r': 30000}
def oname(s): return ('c256-' if s in CS else 'h256-') + s
SEEDS = sorted(int(f.split('_s')[1].split('_')[0]) for f in os.listdir(f'{RES}/runs/kx/export') if f.endswith('_h256.npz'))
ROWS = {}                                                             # {row: {seed: {set: (score, idx)}}}
for sd in SEEDS:
    U = {**dict(np.load(f'{RES}/runs/kx/export/in256_s{sd}_h256.npz')), **dict(np.load(f'{RES}/runs/kx/export/in256_s{sd}_c256.npz'))}
    K = np.load(f'{RES}/runs/in256/knn_s{sd}.npz')
    for s in [ID, VF] + OOD + CS:
        n = oname(s); iu = U[f'idx__{n}']
        for r, z, key, ix in (('Ours', U, 'score', iu), ('kNN term', U, 'content', iu), ('Orientation term', U, 'pose', iu), ('kNN basic', K, 'ct', K[f'idx__{n}'])):
            ROWS.setdefault(r, {}).setdefault(sd, {})[s] = (np.asarray(z[f'{key}__{n}'], np.float64), np.asarray(ix))
    cd = f'{RES}/runs/cfs_l7/s{sd}/in256'
    if os.path.isdir(cd):
        for s in [ID, VF] + OOD + CS:
            C = np.load(f'{cd}/{oname(s)}__scores.npz')
            ROWS.setdefault('CFS', {}).setdefault(sd, {})[s] = (C['diag'].astype(np.float64), C['idx'])
            ROWS.setdefault('CFS kNN', {}).setdefault(sd, {})[s] = (C['knn'].astype(np.float64), C['idx'])
BE.RAW = {k: v for k, v in BE.RAW.items() if k in ('DiffPath', 'GEPC')}       # EigenScore was not run at 256
BE.GEPC_TIER['in256'] = 'cifar10'                                      # the default GEPC configuration (as for ImageNet-64)
def tier256():
    """MSMA*, DiffPath, GEPC of the 256 tier as in baselines_eval.tier_scores (no PF-ODE calibration file exists at 256)"""
    run, idn = 'F_in256', 'hr_' + ID
    mm = BE.Maha().fit(np.log(BE.base(run, idn + '__cal')['s']))
    fitted = {m: mk('in256').fit(BE.load2(raw, 'in256', idn + '__cal')) for m, (raw, mk) in BE.RAW.items()}
    out = {}
    for s in [idn] + ['hr_' + x for x in OOD + [VF]]:
        d = {}; Bz = BE.base(run, s)
        if Bz is not None: d['MSMA*'] = (mm.score(np.log(Bz['s'])), Bz['idx'])
        for m, (raw, _) in BE.RAW.items():
            Z = BE.load2(raw, 'in256', s)
            if Z is not None: d[m] = (fitted[m].score(Z), Z['idx'])
        out[s] = d
    return out, {'DiffPath_grid': {k: str(v) for k, v in fitted['DiffPath'].best.items()}, 'GEPC_kept_levels': [int(l) for l in fitted['GEPC'].kept]}


BLS, info = tier256()
for m in ('MSMA*', 'DiffPath', 'GEPC'):
    for s in [ID, VF] + OOD:
        sc, ix = BLS['hr_' + s][m]; ROWS.setdefault(m, {}).setdefault(0, {})[s] = (np.asarray(sc, np.float64), np.asarray(ix))
ORDER = [r for r in ['Ours', 'kNN term', 'Orientation term', 'kNN basic', 'CFS', 'CFS kNN', 'MSMA*', 'DiffPath', 'GEPC'] if r in ROWS]
HASCS = [r for r in ORDER if CS[0] in ROWS[r][sorted(ROWS[r])[0]]]
# common images of every row (per set) for the comparison table
COM = {}
for s in [ID, VF] + OOD:
    sets = [set(ROWS[r][sd][s][1].tolist()) for r in ORDER for sd in ROWS[r]]; COM[s] = sorted(set.intersection(*sets))
for s in CS:
    sets = [set(ROWS[r][sd][s][1].tolist()) for r in HASCS for sd in ROWS[r]]; COM[s] = sorted(set.intersection(*sets))
ALLID = {s: sorted(set.intersection(*[set(ROWS[r][sd][s][1].tolist()) for r in HASCS for sd in ROWS[r]])) for s in [ID, VF] + OOD + CS}


def wauroc(a, wa, b):
    bs = np.sort(b); f = (np.searchsorted(bs, a, 'left') + 0.5 * (np.searchsorted(bs, a, 'right') - np.searchsorted(bs, a, 'left'))) / len(b)
    return 100 * float(np.sum(wa * f) / np.sum(wa))


def metrics(get, with_cs):
    a = get(ID); out = {}
    for s in OOD + [VF] + (CS if with_cs else []):
        b = get(s); out[s] = dict(auroc=100 * auroc(a, b), fpr95=100 * fpr95(a, b))
    for g, L in (('near', NEAR), ('far', FAR)):
        out[g] = float(np.mean([out[s]['auroc'] for s in L])); out[g + '_fpr95'] = float(np.mean([out[s]['fpr95'] for s in L]))
    if with_cs:
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


OUT = {'seeds': {r: sorted(ROWS[r]) for r in ORDER}, 'n_common': {s: len(v) for s, v in COM.items()}, 'n_all': {s: len(v) for s, v in ALLID.items()},
       'common': {}, 'all': {}, 'fit': info}
for r in ORDER:
    OUT['common'][r] = avg([metrics(lambda s, sd=sd: sub(*ROWS[r][sd][s], COM[s]), False) for sd in sorted(ROWS[r])])
for r in HASCS:
    OUT['all'][r] = avg([metrics(lambda s, sd=sd: sub(*ROWS[r][sd][s], ALLID[s]), True) for sd in sorted(ROWS[r])])
B = 2000; rng = np.random.default_rng(20261031)
def verdict(lo, hi, gap):
    if abs(gap) < 0.05: return 'tie'
    return 'win' if lo > 0 else ('loss' if hi < 0 else 'tie')
Wa = counts(rng, len(COM[ID]), B); Wb = {s: counts(rng, len(COM[s]), B) for s in OOD + [VF]}
def boots(r):
    return {s: np.mean([boot_auroc(sub(*ROWS[r][sd][ID], COM[ID]), sub(*ROWS[r][sd][s], COM[s]), Wa, Wb[s]) for sd in sorted(ROWS[r])], 0) for s in OOD + [VF]}
bo = boots('Ours'); OUT['boot'] = {}
for r in ORDER[1:]:
    br = boots(r); res = {}
    for s in OOD + [VF]:
        d = bo[s] - br[s]; lo, hi = np.percentile(d, [2.5, 97.5]); gap = OUT['common']['Ours'][s]['auroc'] - OUT['common'][r][s]['auroc']
        res[s] = dict(diff=gap, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap))
    for g, L in (('near', NEAR), ('far', FAR)):
        d = np.mean([bo[s] for s in L], 0) - np.mean([br[s] for s in L], 0); lo, hi = np.percentile(d, [2.5, 97.5])
        gap = OUT['common']['Ours'][g] - OUT['common'][r][g]; res[g] = dict(diff=gap, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap))
    vs = [res[s]['verdict'] for s in OOD]; res['wtl'] = [vs.count('win'), vs.count('tie'), vs.count('loss')]
    OUT['boot'][r] = res
# ImageNet-C by severity (all images of the subsample, seeds averaged)
meta = json.load(open(f'{RES}/data/npy/in64_imagenet_c_meta.json')); src = np.load(f'{RES}/data/npy/hr_imagenet_c_src_idx.npy')
sev = np.array([int(meta[i][1]) for i in src]); OUT['imagenet_c'] = {}
for r in [x for x in ('Ours', 'kNN term', 'Orientation term', 'kNN basic', 'CFS') if x in HASCS]:
    bys = {}
    for sd in sorted(ROWS[r]):
        a = ROWS[r][sd][ID][0]; sc, ix = ROWS[r][sd]['imagenet_c']
        for v in range(1, 6): bys.setdefault(v, []).append(100 * auroc(a, sc[sev[ix] == v]))
    OUT['imagenet_c'][r] = {v: float(np.mean(x)) for v, x in bys.items()}
OUT['temp'] = {sd: json.load(open(f'{RES}/runs/kx/rothead/s{sd}/in256_k0_gqR/temp.json')) for sd in SEEDS}
json.dump(OUT, open(f'{HERE}/in256_ana.json', 'w'), indent=1)
print('seeds', OUT['seeds']); print('common images', OUT['n_common']); print('all images', OUT['n_all'])
print('== common images: SSB NINCO | near | iNat Tex OIO | far | vflip | FPR95 near far | W/T/L (5 OOD sets) near far')
for r in ORDER:
    M = OUT['common'][r]
    ex = '' if r == 'Ours' else f"  {OUT['boot'][r]['wtl']} {OUT['boot'][r]['near']['verdict']} {OUT['boot'][r]['far']['verdict']} vflip {OUT['boot'][r][VF]['verdict']}"
    print(f"{r:17s}" + ' '.join(f"{M[s]['auroc']:6.1f}" for s in NEAR) + f" | {M['near']:5.1f} | " + ' '.join(f"{M[s]['auroc']:6.1f}" for s in FAR) +
          f" | {M['far']:5.1f} | {M[VF]['auroc']:5.1f} | {M['near_fpr95']:5.1f} {M['far_fpr95']:5.1f}{ex}")
print('== all images of our runs (1,000 per set): near far vflip | V2 C R | FS near far')
for r in HASCS:
    M = OUT['all'][r]
    print(f"{r:17s} {M['near']:5.1f} {M['far']:5.1f} {M[VF]['auroc']:5.1f} | " + ' '.join(f"{M[s]['auroc']:5.1f}" for s in CS) + f" | {M['FS-near']:5.1f} {M['FS-far']:5.1f}")
print('ImageNet-C by severity', {r: [round(v[k], 1) for k in range(1, 6)] for r, v in OUT['imagenet_c'].items()})
print('temperatures', OUT['temp'])
