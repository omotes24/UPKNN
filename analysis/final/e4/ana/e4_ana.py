# e4_ana.py -- the tiers of PROTOCOL_E4.md (written before the results; analysis written before the results were read).
#   A/B (chest, path): Upright kNN (seeds 0-2), its kNN and orientation terms, kNN (basic), CFS diag / kNN heads (seeds 0-2), NLL, MSMA*,
#   DiffPath (fitted on the '__cal' run of 1,000 ID training images outside the bank).  AUROC (ID positive) per set and per group;
#   bootstrap of Upright kNN against every row (2,000 replicates, rng 20261031, the paper's verdict rule) per set and on the group means.
#   C (c10n4, c10x4, c10r25, c10r50, c10r100): Upright kNN and its terms, kNN (basic), NLL of the generator, and NLL + log P_T(up|x)
#   (the head's orientation term, mean over seeds 0-2): natural-OOD mean (nine sets) and ID vs vertical flip.
#   usage: python e4_ana.py <results dir (contains runs/, logs/)>        -> e4_ana.json
import os, sys, json, re
import numpy as np
SCR = os.environ.get('UPKNN_ANALYSIS', os.path.expanduser('~/upknn/analysis'))
HERE = os.path.dirname(os.path.abspath(__file__)); RES = sys.argv[1] if len(sys.argv) > 1 else f'{SCR}/final/e4/res'
sys.path.insert(0, f'{SCR}/final/ana'); sys.path.insert(0, f'{SCR}/eu_build/Euphonium/analysis')
os.environ.setdefault('EUPHONIUM_INPUTS', f'{SCR}/eu_build/inputs_full')
from common import auroc, fpr95
from boot import counts, boot_auroc
import baselines_eval as BE
SEEDS = (0, 1, 2); B = 2000
TF = ('vflip', 'rot90', 'rot180', 'hflip')
GROUPS = {
    'chest': dict(id='g32_chest_test', modality=['g32_organa_test', 'g32_path_gray_test'], source=['g32_pneumonia_all'],
                  natural=['g32_cifar10_gray', 'g32_svhn_gray', 'g32_mnist_test', 'g32_fmnist_test']),
    'path': dict(id='path_test', modality=['blood_all', 'derma_all', 'chest_rgb_test'], source=[],
                 natural=['cifar10_test', 'cifar100_test', 'svhn_test', 'celeba_test', 'textures', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val'])}
C10 = ['c10n4', 'c10x4', 'c10r25', 'c10r50', 'c10r100']
NAT9 = ['cifar100_test', 'svhn_test', 'celeba_test', 'textures', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val']


def verdict(lo, hi, gap):
    if abs(gap) < 0.05: return 'tie'
    return 'win' if lo > 0 else ('loss' if hi < 0 else 'tie')


def ld(p): return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def ours_rows(tag, sets):
    """{row: {seed: {set: score}}} for Upright kNN, its two terms and kNN (basic); higher = more ID"""
    R = {}
    for sd in SEEDS:
        U = ld(f'{RES}/runs/kx/export/{tag}_s{sd}_e4.npz'); K = ld(f'{RES}/runs/e4/knn_{tag}_s{sd}.npz')
        if U is None or K is None: continue
        for s in sets:
            n = f'e4-{s}'
            for r, z, key in (('Ours', U, 'score'), ('kNN term', U, 'content'), ('Orientation term', U, 'pose'), ('kNN basic', K, 'ct')):
                R.setdefault(r, {}).setdefault(sd, {})[s] = np.asarray(z[f'{key}__{n}'], np.float64)
            assert (U[f'idx__{n}'] == K[f'idx__{n}']).all()
    return R


def single(v): return {0: v}


def tier_ab(tag):
    G = GROUPS[tag]; sets = [G['id']] + [f"{G['id']}__{t}" for t in TF] + G['modality'] + G['source'] + G['natural']
    R = ours_rows(tag, sets)
    if 'Ours' not in R: return None
    # CFS (3 seeds)
    for sd in SEEDS:
        for s in sets:
            z = ld(f'{RES}/runs/cfs_l7/s{sd}/{tag}/e4-{s}__scores.npz')
            if z is None: continue
            R.setdefault('CFS', {}).setdefault(sd, {})[s] = np.asarray(z['diag'], np.float64)      # stored ID-high (cfs_score.py)
            R.setdefault('CFS-kNN', {}).setdefault(sd, {})[s] = np.asarray(z['knn'], np.float64)
    # NLL, MSMA*, DiffPath (one run each; fitted on the '__cal' run)
    cal_b = ld(f'{RES}/runs/base/E4_{tag}/{G["id"]}__cal.npz'); cal_d = ld(f'{RES}/runs/base2/diffpath_E4_{tag}/{G["id"]}__cal.npz')
    mm = BE.Maha().fit(np.log(cal_b['s'])) if cal_b is not None else None
    dpm = BE.DiffPath6D().fit({k: cal_d[k] for k in cal_d.files}) if cal_d is not None else None
    for s in sets:
        z = ld(f'{RES}/runs/nll/E4_{tag}/{s}.npz')
        if z is not None: R.setdefault('NLL', {}).setdefault(0, {})[s] = np.asarray(z['ll'], np.float64)
        z = ld(f'{RES}/runs/base/E4_{tag}/{s}.npz')
        if z is not None and mm is not None: R.setdefault('MSMA*', {}).setdefault(0, {})[s] = mm.score(np.log(z['s']))
        z = ld(f'{RES}/runs/base2/diffpath_E4_{tag}/{s}.npz')
        if z is not None and dpm is not None: R.setdefault('DiffPath', {}).setdefault(0, {})[s] = dpm.score({k: z[k] for k in z.files})
    return compare(R, G['id'], [s for s in sets if s != G['id']], {'modality': G['modality'], 'natural': G['natural']})


def compare(R, ID, others, groups):
    rows = [r for r in R if all(all(s in R[r][sd] for s in [ID] + others) for sd in R[r])]
    OUT = {'rows': {}, 'boot': {}}
    for r in rows:
        sds = sorted(R[r]); per = {s: float(np.mean([100 * auroc(R[r][sd][ID], R[r][sd][s]) for sd in sds])) for s in others}
        OUT['rows'][r] = dict(per_set=per, seeds=len(sds), **{g: float(np.mean([per[s] for s in v])) for g, v in groups.items()},
                              fpr95={s: float(np.mean([100 * fpr95(R[r][sd][ID], R[r][sd][s]) for sd in sds])) for s in others})
    rng = np.random.default_rng(20261031)
    Wa = counts(rng, len(R['Ours'][0][ID]), B); Wb = {s: counts(rng, len(R['Ours'][0][s]), B) for s in others}
    def boots(r): return {s: np.mean([boot_auroc(R[r][sd][ID], R[r][sd][s], Wa, Wb[s]) for sd in sorted(R[r])], 0) for s in others}
    bo = boots('Ours')
    for r in [x for x in rows if x != 'Ours']:
        br = boots(r); res = {}
        for s in others:
            d = bo[s] - br[s]; lo, hi = np.percentile(d, [2.5, 97.5]); gap = OUT['rows']['Ours']['per_set'][s] - OUT['rows'][r]['per_set'][s]
            res[s] = dict(diff=gap, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap))
        for g, v in groups.items():
            d = np.mean([bo[s] for s in v], 0) - np.mean([br[s] for s in v], 0); lo, hi = np.percentile(d, [2.5, 97.5])
            gap = OUT['rows']['Ours'][g] - OUT['rows'][r][g]; res[g] = dict(diff=gap, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap))
        vs = [res[s]['verdict'] for s in others]; res['wtl'] = [vs.count('win'), vs.count('tie'), vs.count('loss')]
        OUT['boot'][r] = res
    return OUT


def heads(tag):
    out = {}
    for sd in SEEDS:
        p = f'{RES}/logs/e4/head_{tag}_s{sd}.log'
        if not os.path.exists(p): continue
        t = open(p).read(); acc = re.findall(rf'{tag} mlp epoch \d+ loss [0-9.]+ val acc ([0-9.]+)', t); T = re.findall(r"'mlp': ([0-9.]+)", t)
        if acc and T: out[sd] = dict(acc=float(acc[-1]), T=float(T[-1]))
    return out


def tier_c(tag):
    sets = ['cifar10_test', 'cifar10_test__vflip'] + NAT9
    R = ours_rows(tag, sets)
    if 'Ours' not in R: return None
    nl = {s: ld(f'{RES}/runs/nll/E4_{tag}/{s}.npz') for s in sets}
    if all(v is not None for v in nl.values()):
        R['NLL'] = {0: {s: np.asarray(nl[s]['ll'], np.float64) for s in sets}}
        pose = {s: np.mean([R['Orientation term'][sd][s] for sd in R['Orientation term']], 0) for s in sets}
        R['NLL + log P(up|x)'] = {0: {s: R['NLL'][0][s] + pose[s] for s in sets}}
    return compare(R, 'cifar10_test', sets[1:], {'natural': NAT9, 'flip': ['cifar10_test__vflip']})


OUT = {}
for tag in ('chest', 'path'):
    r = tier_ab(tag)
    if r: r['heads'] = heads(tag); OUT[tag] = r
for tag in C10:
    r = tier_c(tag)
    if r: r['heads'] = heads(tag); OUT[tag] = r
json.dump(OUT, open(f'{HERE}/e4_ana.json', 'w'), indent=1)
for tag, v in OUT.items():
    print(f'== {tag}  heads {v["heads"]}')
    for r, x in v['rows'].items():
        extra = {k: round(x[k], 1) for k in ('modality', 'natural', 'flip') if k in x}
        tfs = {t: round(x['per_set'][s], 1) for s in x['per_set'] for t in TF if s.endswith('__' + t)}
        b = v['boot'].get(r, {}); vv = {k: b[k]['verdict'] for k in ('modality', 'natural', 'flip') if k in b}
        print(f'  {r:20s} {extra} {tfs} W/T/L {b.get("wtl")} {vv}')
