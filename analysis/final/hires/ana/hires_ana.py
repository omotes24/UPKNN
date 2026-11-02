# hires_ana.py -- the resolution-controlled ImageNet tiers (PROTOCOL_HIRES.md, PROTOCOL_HIRES_DIT.md): in256c (ADM 256, pixels),
#   e512 (EDM2-XS-512 uncond, latents), dit256 (DiT-XL/2 256, latents).  Every image passed through the same 224-px bottleneck, so
#   SSB-hard and ImageNet-C are included.  Rows per tier: Ours (3 seeds), kNN term, orientation term, kNN (basic) (3 seeds); CFS diag / kNN
#   (3 seeds; in256c, e512); MSMA*, DiffPath, GEPC (in256c only; fits on 1,000 ID calibration images; 500 images per set).
#   Comparison on the images every row of the tier has; bootstrap 2000, rng 20261031, verdict as everywhere; full spectrum with ID + csID
#   weighted by the OpenOOD set sizes.  Also Ours across the three tiers on their common images.   usage: python hires_ana.py <results dir>
import os, sys, json
import numpy as np
SCR = os.environ.get('UPKNN_ANALYSIS', os.path.expanduser('~/upknn/analysis'))
HERE = os.path.dirname(os.path.abspath(__file__)); RES = sys.argv[1] if len(sys.argv) > 1 else f'{SCR}/final/hires/res'
os.environ['EUPHONIUM_INPUTS'] = f'{RES}'                                  # base/F_in256c, base2/<m>_in256c under the results tree
sys.path.insert(0, f'{SCR}/eu_build/Euphonium/analysis'); sys.path.insert(0, f'{SCR}/final/ana')
import common as CM
CM.INPUTS = f'{RES}/runs'
import baselines_eval as BE
BE.INPUTS = f'{RES}/runs'
from common import sub, auroc, fpr95
from boot import counts, boot_auroc
NEAR = ['ssb_hard', 'ninco']; FAR = ['inaturalist', 'textures', 'openimage_o']; OOD = NEAR + FAR; VF = 'imagenet_test__vflip'
CS = ['imagenet_v2', 'imagenet_c', 'imagenet_r']; ID = 'imagenet_test'; ALLS = [ID, VF] + OOD + CS
NOPENOOD = {ID: 45000, 'imagenet_v2': 10000, 'imagenet_c': 10000, 'imagenet_r': 30000}
PFX = {'in256c': 'q256-', 'e512': 'q512-', 'dit256': 'qdit-'}; LZ = {'e512': 'lz512', 'dit256': 'lzdit'}
SEEDS = (0, 1, 2)


def src_map(tag, s):
    """positions in the tier's arrays -> original image indices (latent tiers store only the selected images)"""
    if tag == 'in256c': return None
    return np.load(f'{RES}/data/npy/{LZ[tag]}_{s}_src_idx.npy')


def rows_for(tag):
    R = {}
    for sd in SEEDS:
        fx = [f'{RES}/runs/kx/export/{tag}_s{sd}_{PFX[tag].replace("-", "")}.npz']
        if not os.path.exists(fx[0]): continue
        U = np.load(fx[0]); K = np.load(f'{RES}/runs/hires/knn_{tag}_s{sd}.npz')
        for s in ALLS:
            n = PFX[tag] + s; m = src_map(tag, s)
            orig = lambda ix: np.asarray(ix) if m is None else m[np.asarray(ix)]
            for r, z, key in (('Ours', U, 'score'), ('kNN term', U, 'content'), ('Orientation term', U, 'pose'), ('kNN basic', K, 'ct')):
                R.setdefault(r, {}).setdefault(sd, {})[s] = (np.asarray(z[f'{key}__{n}'], np.float64), orig(z[f'idx__{n}']))
        cd = f'{RES}/runs/cfs_l7/s{sd}/{tag}'
        if os.path.isdir(cd) and all(os.path.exists(f'{cd}/{PFX[tag]}{s}__scores.npz') for s in ALLS):
            for s in ALLS:
                C = np.load(f'{cd}/{PFX[tag]}{s}__scores.npz'); m = src_map(tag, s)
                ix = np.asarray(C['idx']) if m is None else m[np.asarray(C['idx'])]
                R.setdefault('CFS', {}).setdefault(sd, {})[s] = (C['diag'].astype(np.float64), ix)
                R.setdefault('CFS kNN', {}).setdefault(sd, {})[s] = (C['knn'].astype(np.float64), ix)
    if tag == 'in256c':
        BE.RAW = {k: v for k, v in BE.RAW.items() if k in ('DiffPath', 'GEPC')}; BE.GEPC_TIER['in256c'] = 'cifar10'
        run, idn = 'F_in256c', 'hc_imagenet_test'
        have = lambda raw: os.path.exists(f'{RES}/runs/base2/{raw}_in256c/{idn}__cal.npz')
        calB = BE.base(run, idn + '__cal'); mm = BE.Maha().fit(np.log(calB['s'])) if calB is not None else None
        fitted = {m: mk('in256c').fit(BE.load2(raw, 'in256c', idn + '__cal')) for m, (raw, mk) in BE.RAW.items() if have(raw)}
        for s in ALLS:
            name = f'hc_{s}' if s != VF else 'hc_imagenet_test__vflip'
            Bz = BE.base(run, name)
            if mm is not None and Bz is not None: R.setdefault('MSMA*', {}).setdefault(0, {})[s] = (mm.score(np.log(Bz['s'])), np.asarray(Bz['idx']))
            for m, (raw, _) in BE.RAW.items():
                Z = BE.load2(raw, 'in256c', name) if m in fitted else None
                if Z is not None: R.setdefault(m, {}).setdefault(0, {})[s] = (fitted[m].score(Z), np.asarray(Z['idx']))
        for m in ('MSMA*', 'DiffPath', 'GEPC'):                              # keep a baseline only if it has every set
            if m in R and not all(s in R[m][0] for s in ALLS): del R[m]
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


OUT = {}
ALLROWS = {}
for tag in ('in256c', 'e512', 'dit256'):
    R = rows_for(tag)
    if 'Ours' not in R: print(tag, 'no results yet'); continue
    ALLROWS[tag] = R
    order = [r for r in ['Ours', 'kNN term', 'Orientation term', 'kNN basic', 'CFS', 'CFS kNN', 'MSMA*', 'DiffPath', 'GEPC'] if r in R]
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
        p = f'{RES}/runs/kx/rothead/s{sd}/{tag}_k0_gqR/temp.json'
        if os.path.exists(p): T['temp'][sd] = json.load(open(p))
    OUT[tag] = T
    print(f'== {tag}: common images', T['n_common'])
    print('   SSB NINCO | near | iNat Tex OIO | far | flip | V2 C R | FS near far | W/T/L')
    for r in order:
        M = T['common'][r]; ex = '' if r == 'Ours' else f"  {T['boot'][r]['wtl']} near {T['boot'][r]['near']['verdict']} far {T['boot'][r]['far']['verdict']}"
        print(f"{r:17s}" + ' '.join(f"{M[s]['auroc']:5.1f}" for s in NEAR) + f" | {M['near']:5.1f} | " + ' '.join(f"{M[s]['auroc']:5.1f}" for s in FAR) +
              f" | {M['far']:5.1f} | {M[VF]['auroc']:5.1f} | " + ' '.join(f"{M[s]['auroc']:5.1f}" for s in CS) + f" | {M['FS-near']:5.1f} {M['FS-far']:5.1f}{ex}")
# Ours across tiers on the images all tiers share
if len(ALLROWS) > 1:
    tags = list(ALLROWS)
    COMX = {s: sorted(set.intersection(*[set(ALLROWS[t]['Ours'][sd][s][1].tolist()) for t in tags for sd in ALLROWS[t]['Ours']])) for s in ALLS}
    OUT['cross'] = {t: avg([metrics(lambda s, sd=sd, t=t: sub(*ALLROWS[t]['Ours'][sd][s], COMX[s])) for sd in sorted(ALLROWS[t]['Ours'])]) for t in tags}
    print('== Ours across tiers (common images):', {t: (round(v['near'], 1), round(v['far'], 1), round(v[VF]['auroc'], 1)) for t, v in OUT['cross'].items()})
json.dump(OUT, open(f'{HERE}/hires_ana.json', 'w'), indent=1)
