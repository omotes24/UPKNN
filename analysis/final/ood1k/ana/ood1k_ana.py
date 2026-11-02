# ood1k_ana.py -- the OpenOOD v1.5 ImageNet-1K benchmark at 64x64 (prep_ood1k.py; frozen ImageNet-64 detectors, no refitting).
#   Sets: ID = ImageNet val (OpenOOD test list), near-OOD = SSB-hard, NINCO, far-OOD = iNaturalist, Textures, OpenImage-O,
#   covariate-shifted ID (csID) = ImageNet-V2, -C, -R; images used earlier for development / selection / final evaluation excluded.
#   Rows on the full sets (noise seeds 0-2): Ours (Upright kNN), its two terms, Vertical Limit, kNN (basic: PROTOCOL_CT), CFS (diag / kNN
#   heads, lambda 7), the rotation classifier (RotCls4, one seed).  Comparison rows on a fixed subsample of 1,000 images per set
#   (runs/ood1k/s1k, RandomState(20261102)): NLL, IC (PNG, JPEG2000), TT, MSMA*, DiffPath, GEPC, EigenScore (analysis/baselines_eval.py,
#   fits on the ID development images of the main evaluation, unchanged).  Every row is ALSO evaluated on the same 1,000 images; the
#   comparison table uses these common images, and the bootstrap (2000 replicates, rng 20261031, ID resampled once, OOD per set, verdict
#   win/loss if the 95% CI of AUROC(Ours) - AUROC(row) excludes 0 and |gap| >= 0.05) runs on them.
#   Full spectrum (OpenOOD v1.5 FSOOD): ID and csID together are the positive class; on the 1,000-image subsample every image carries the
#   weight (full-set size) / 1000 of its set, so that the sets enter in the proportions of the full benchmark.
#   output: ood1k_ana.json (+ printed tables)
import os, sys, json, glob
import numpy as np
SCR = os.environ.get('UPKNN_ANALYSIS', os.path.expanduser('~/upknn/analysis'))
RES = f'{SCR}/final/ood1k/res'; HERE = os.path.dirname(os.path.abspath(__file__))
SRC = f'{SCR}/eu_build/inputs_full'; INP = f'{SCR}/final/ood1k/inputs'
ID = 'in64_imagenet_test'
NEAR = ['in64_ssb_hard', 'in64_ninco']; FAR = ['in64_inaturalist', 'in64_textures', 'in64_openimage_o']
CS = ['in64_imagenet_v2', 'in64_imagenet_c', 'in64_imagenet_r']; OOD = NEAR + FAR; ALLS = [ID] + OOD + CS


def link(src, dst):
    assert os.path.exists(src), src
    if os.path.lexists(dst): os.remove(dst)
    os.symlink(os.path.realpath(src), dst)


# ---- 1. inputs tree for baselines_eval.py: calibration files of the main evaluation + the per-set statistics of the 1,000-image subsample
for kind, src, dst in (('nll', f'{RES}/runs/nll/F_in64_ood1k', f'{INP}/nll/F_in64'), ('base', f'{RES}/runs/base/F_in64_ood1k', f'{INP}/base/F_in64')):
    os.makedirs(dst, exist_ok=True); link(f'{SRC}/{kind}/F_in64/{ID}__cal.npz', f'{dst}/{ID}__cal.npz')
    for s in ALLS: link(f'{src}/{s}.npz', f'{dst}/{s}.npz')
for raw in ('diffpath', 'gepc', 'eigen'):
    dst = f'{INP}/base2/{raw}_in64'; os.makedirs(dst, exist_ok=True); link(f'{SRC}/base2/{raw}_in64/{ID}__cal.npz', f'{dst}/{ID}__cal.npz')
    for s in ALLS: link(f'{RES}/runs/base2/{raw}_in64_ood1k/{s}.npz', f'{dst}/{s}.npz')
os.makedirs(f'{INP}/complexity', exist_ok=True)
for s in ALLS:
    c = f'{RES}/data/complexity/{s}.npz'
    link(c if os.path.exists(c) else f'{SRC}/complexity/{s}.npz', f'{INP}/complexity/{s}.npz')
os.environ['EUPHONIUM_INPUTS'] = INP
sys.path.insert(0, f'{SCR}/eu_build/Euphonium/analysis'); sys.path.insert(0, f'{SCR}/final/ana')
import baselines_eval as BE
from common import sub, auroc, fpr95
from boot import counts, boot_auroc

S1K = json.load(open(f'{RES}/runs/ood1k/s1k/s1k_idx.json'))['in64']
K1 = {s: S1K['__id__' if s == ID else s] for s in ALLS}
SETS = {r[1]: r[2] for r in json.load(open(f'{RES}/runs/ok/sets_in64.json')) if r[0].startswith('ood1k-')}
NFULL = {s: len(SETS[s]) for s in ALLS}
SEEDS = (0, 1, 2)

# ---- 2. per-image scores: {row: {seed: {set: (score, idx)}}}
ROWS = {}
for sd in SEEDS:
    U = np.load(f'{RES}/runs/kx/export/in64_s{sd}_ood1k.npz'); V = np.load(f'{RES}/runs/vl/export/in64_s{sd}_ood1k.npz')
    K = np.load(f'{RES}/runs/ood1k/knn_s{sd}.npz')
    for s in ALLS:
        n = f'ood1k-{s}'; iu = U[f'idx__{n}']
        for r, z, key, ix in (('Ours', U, 'score', iu), ('kNN term', U, 'content', iu), ('Orientation term', U, 'pose', iu),
                              ('VL', V, 'score', V[f'idx__{n}']), ('kNN basic', K, 'ct', K[f'idx__{n}'])):
            ROWS.setdefault(r, {}).setdefault(sd, {})[s] = (z[f'{key}__{n}'].astype(np.float64), np.asarray(ix))
        C = np.load(f'{RES}/runs/cfs_l7/s{sd}/in64/ood1k-{s}__scores.npz')
        ROWS.setdefault('CFS', {}).setdefault(sd, {})[s] = (C['diag'].astype(np.float64), C['idx'])
        ROWS.setdefault('CFS kNN', {}).setdefault(sd, {})[s] = (C['knn'].astype(np.float64), C['idx'])
for s in ALLS:
    z = np.load(f'{RES}/runs/rotcls/in64/scores_ood1k/{s}.npz')
    ROWS.setdefault('RotCls4', {}).setdefault(0, {})[s] = (z['rot4'].astype(np.float64), z['idx'])
BLS, info = BE.tier_scores('in64', dict(id=ID, ood=OOD + CS))
BLN = ['NLL', 'IC-png', 'IC-j2k', 'TT', 'MSMA*', 'DiffPath', 'GEPC', 'EigenScore']
for m in BLN:
    for s in ALLS:
        sc, ix = BLS[s][m]; ROWS.setdefault(m, {}).setdefault(0, {})[s] = (np.asarray(sc, np.float64), np.asarray(ix))
FULLROWS = ['Ours', 'kNN term', 'Orientation term', 'VL', 'kNN basic', 'CFS', 'CFS kNN', 'RotCls4']
ORDER = FULLROWS + BLN


def wauroc(a, wa, b):
    """AUROC with weighted positives a (weights wa) against unweighted negatives b (ties count 1/2), in %"""
    bs = np.sort(b); f = (np.searchsorted(bs, a, 'left') + 0.5 * (np.searchsorted(bs, a, 'right') - np.searchsorted(bs, a, 'left'))) / len(b)
    return 100 * float(np.sum(wa * f) / np.sum(wa))


def metrics(get, sets_k, weights):
    """get(set) -> scores; per-set AUROC / FPR95 (ID vs set) and the full-spectrum AUROC of ID + csID (weighted) vs each OOD set"""
    a = get(ID); out = {}
    for s in OOD + CS:
        b = get(s); out[s] = dict(auroc=100 * auroc(a, b), fpr95=100 * fpr95(a, b))
    pos = np.concatenate([get(s) for s in [ID] + CS]); w = np.concatenate([np.full(len(get(s)), weights(s)) for s in [ID] + CS])
    for s in OOD: out['FS-' + s] = dict(auroc=wauroc(pos, w, get(s)))
    for g, L in (('near', NEAR), ('far', FAR)):
        out[g] = float(np.mean([out[s]['auroc'] for s in L])); out['FS-' + g] = float(np.mean([out['FS-' + s]['auroc'] for s in L]))
        out[g + '_fpr95'] = float(np.mean([out[s]['fpr95'] for s in L]))
    return out


def avg(dicts):
    """mean over seeds of nested metric dicts"""
    if len(dicts) == 1: return dicts[0]
    o = {}
    for k, v in dicts[0].items():
        o[k] = {kk: float(np.mean([d[k][kk] for d in dicts])) for kk in v} if isinstance(v, dict) else float(np.mean([d[k] for d in dicts]))
    return o


OUT = {'n_full': NFULL, 'n_sub': {s: len(K1[s]) for s in ALLS}, 'full': {}, 'common': {}, 'fit': info}
for r in ORDER:
    sds = sorted(ROWS[r])
    OUT['common'][r] = avg([metrics(lambda s, sd=sd: sub(*ROWS[r][sd][s], K1[s]), K1, lambda s: NFULL[s] / len(K1[s])) for sd in sds])
    if r in FULLROWS:
        OUT['full'][r] = avg([metrics(lambda s, sd=sd: ROWS[r][sd][s][0], None, lambda s: 1.0) for sd in sds])
for s in ALLS: assert all(len(sub(*ROWS[r][sd][s], K1[s])) == len(K1[s]) for r in ORDER for sd in ROWS[r]), s   # every row has every subsample image

# ---- 3. bootstrap on the common 1,000 images: Ours (3 seeds) against every row, per OOD set and for the near / far means
B = 2000; rng = np.random.default_rng(20261031)
def verdict(lo, hi, gap):
    if abs(gap) < 0.05: return 'tie'
    return 'win' if lo > 0 else ('loss' if hi < 0 else 'tie')
Wa = counts(rng, len(K1[ID]), B); Wb = {s: counts(rng, len(K1[s]), B) for s in OOD}
def boots(r):
    return {s: np.mean([boot_auroc(sub(*ROWS[r][sd][ID], K1[ID]), sub(*ROWS[r][sd][s], K1[s]), Wa, Wb[s]) for sd in sorted(ROWS[r])], 0) for s in OOD}
bo = boots('Ours'); OUT['boot'] = {}
for r in ORDER[1:]:
    br = boots(r); res = {}
    for s in OOD:
        d = bo[s] - br[s]; lo, hi = np.percentile(d, [2.5, 97.5]); gap = OUT['common']['Ours'][s]['auroc'] - OUT['common'][r][s]['auroc']
        res[s] = dict(diff=gap, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap))
    for g, L in (('near', NEAR), ('far', FAR)):
        d = np.mean([bo[s] for s in L], 0) - np.mean([br[s] for s in L], 0); lo, hi = np.percentile(d, [2.5, 97.5])
        gap = OUT['common']['Ours'][g] - OUT['common'][r][g]; res[g] = dict(diff=gap, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap))
    vs = [res[s]['verdict'] for s in OOD]; res['wtl'] = [vs.count('win'), vs.count('tie'), vs.count('loss')]
    OUT['boot'][r] = res

# ---- 4. ImageNet-C by severity and corruption (full sets, seeds 0-2): Ours and its two terms
meta = json.load(open(f'{RES}/data/npy/in64_imagenet_c_meta.json'))
corr = np.array([m[0] for m in meta]); sev = np.array([int(m[1]) for m in meta])
OUT['imagenet_c'] = {}
for r in ('Ours', 'kNN term', 'Orientation term', 'kNN basic', 'CFS'):
    bysev, bycor = {}, {}
    for sd in SEEDS:
        a = ROWS[r][sd][ID][0]; sc, ix = ROWS[r][sd]['in64_imagenet_c']
        for v in range(1, 6): bysev.setdefault(v, []).append(100 * auroc(a, sc[sev[ix] == v]))
        for c in sorted(set(corr)): bycor.setdefault(c, []).append(100 * auroc(a, sc[corr[ix] == c]))
    OUT['imagenet_c'][r] = dict(severity={v: float(np.mean(x)) for v, x in bysev.items()}, corruption={c: float(np.mean(x)) for c, x in bycor.items()})

json.dump(OUT, open(f'{HERE}/ood1k_ana.json', 'w'), indent=1)
# ---- printed tables
def line(r, M, extra=''):
    return f"{r:17s} " + ' '.join(f"{M[s]['auroc']:6.1f}" for s in NEAR) + f" | {M['near']:6.1f} | " + ' '.join(f"{M[s]['auroc']:6.1f}" for s in FAR) + \
           f" | {M['far']:6.1f} | " + ' '.join(f"{M[s]['auroc']:6.1f}" for s in CS) + f" | FS {M['FS-near']:6.1f} {M['FS-far']:6.1f} | FPR {M['near_fpr95']:5.1f} {M['far_fpr95']:5.1f}{extra}"
print('sizes', NFULL)
print('== common 1,000 images per set (AUROC %): SSB NINCO | near | iNat Tex OIO | far | V2 C R | FS near far | FPR95 near far | W/T/L')
for r in ORDER:
    print(line(r, OUT['common'][r], '' if r == 'Ours' else f"  W/T/L {OUT['boot'][r]['wtl']}  near {OUT['boot'][r]['near']['verdict']} far {OUT['boot'][r]['far']['verdict']}"))
print('== full sets')
for r in FULLROWS: print(line(r, OUT['full'][r]))
print('== ImageNet-C by severity (AUROC ID vs C_s)')
for r, v in OUT['imagenet_c'].items(): print(f'{r:17s}', ' '.join(f'{v["severity"][k]:6.1f}' for k in range(1, 6)))
print('fit', info)
