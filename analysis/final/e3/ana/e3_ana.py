# e3_ana.py -- the EuroSAT tier (PROTOCOL_E3.md): Upright kNN on aerial images, noise seeds 0-2.
#   Rows: Ours, kNN term, orientation term, kNN (basic).  AUROC (ID positive) per OOD set, mean over the ten natural OOD sets,
#   ID vs its vertical flips and its 90-degree rotations; bootstrap of Ours against each row (2000 replicates, rng 20261031,
#   verdict as everywhere) over the ten OOD sets; head temperature / held-out accuracy from the logs.   output: e3_ana.json
import os, sys, json, re, glob
import numpy as np
SCR = os.environ.get('UPKNN_ANALYSIS', os.path.expanduser('~/upknn/analysis'))
HERE = os.path.dirname(os.path.abspath(__file__)); RES = f'{SCR}/final/e3/res'
sys.path.insert(0, f'{SCR}/final/ana'); sys.path.insert(0, f'{SCR}/eu_build/Euphonium/analysis')
os.environ.setdefault('EUPHONIUM_INPUTS', f'{SCR}/eu_build/inputs_full')
from common import auroc, fpr95
from boot import counts, boot_auroc
ID = 'eurosat_test'; OOD = ['cifar10_test', 'cifar100_test', 'svhn_test', 'celeba_test', 'textures', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val']
TF = ['eurosat_test__vflip', 'eurosat_test__rot90']; SEEDS = (0, 1, 2)
R = {}
for sd in SEEDS:
    U = np.load(f'{RES}/runs/kx/export/eurosat_s{sd}_e3.npz'); K = np.load(f'{RES}/runs/e3/knn_s{sd}.npz')
    for s in [ID] + OOD + TF:
        n = f'e3-{s}'
        for r, z, key in (('Ours', U, 'score'), ('kNN term', U, 'content'), ('Orientation term', U, 'pose'), ('kNN basic', K, 'ct')):
            R.setdefault(r, {}).setdefault(sd, {})[s] = np.asarray(z[f'{key}__{n}'], np.float64)
    assert all((U[f'idx__e3-{s}'] == K[f'idx__e3-{s}']).all() for s in [ID] + OOD + TF)
OUT = {'rows': {}, 'boot': {}}
for r in R:
    per = {s: [100 * auroc(R[r][sd][ID], R[r][sd][s]) for sd in SEEDS] for s in OOD + TF}
    OUT['rows'][r] = {'per_set': {s: float(np.mean(v)) for s, v in per.items()}, 'mean': float(np.mean([np.mean(per[s]) for s in OOD])),
                      'mean_sd_seeds': float(np.std([np.mean([per[s][i] for s in OOD]) for i in range(len(SEEDS))])),
                      'vflip': float(np.mean(per[TF[0]])), 'rot90': float(np.mean(per[TF[1]])),
                      'fpr95': float(np.mean([np.mean([100 * fpr95(R[r][sd][ID], R[r][sd][s]) for sd in SEEDS]) for s in OOD]))}
rng = np.random.default_rng(20261031); B = 2000
Wa = counts(rng, len(R['Ours'][0][ID]), B); Wb = {s: counts(rng, len(R['Ours'][0][s]), B) for s in OOD + TF}
def boots(r): return {s: np.mean([boot_auroc(R[r][sd][ID], R[r][sd][s], Wa, Wb[s]) for sd in SEEDS], 0) for s in OOD + TF}
def verdict(lo, hi, gap):
    if abs(gap) < 0.05: return 'tie'
    return 'win' if lo > 0 else ('loss' if hi < 0 else 'tie')
bo = boots('Ours')
for r in [x for x in R if x != 'Ours']:
    br = boots(r); res = {}
    for s in OOD + TF:
        d = bo[s] - br[s]; lo, hi = np.percentile(d, [2.5, 97.5]); gap = OUT['rows']['Ours']['per_set'][s] - OUT['rows'][r]['per_set'][s]
        res[s] = dict(diff=gap, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap))
    d = np.mean([bo[s] for s in OOD], 0) - np.mean([br[s] for s in OOD], 0); lo, hi = np.percentile(d, [2.5, 97.5])
    gap = OUT['rows']['Ours']['mean'] - OUT['rows'][r]['mean']; res['mean'] = dict(diff=gap, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap))
    vs = [res[s]['verdict'] for s in OOD]; res['wtl'] = [vs.count('win'), vs.count('tie'), vs.count('loss')]
    OUT['boot'][r] = res
heads = {}
for sd in SEEDS:
    t = open(f'{RES}/logs/e3/head_s{sd}.log').read()
    acc = re.findall(r'eurosat mlp epoch 7 loss [0-9.]+ val acc ([0-9.]+)', t); T = re.findall(r"'mlp': ([0-9.]+)", t)
    heads[sd] = dict(acc=float(acc[-1]), T=float(T[-1]))
OUT['heads'] = heads
json.dump(OUT, open(f'{HERE}/e3_ana.json', 'w'), indent=1)
print('heads', heads)
for r, v in OUT['rows'].items():
    ex = '' if r == 'Ours' else f"  W/T/L {OUT['boot'][r]['wtl']} mean {OUT['boot'][r]['mean']['verdict']} vflip {OUT['boot'][r][TF[0]]['verdict']} rot90 {OUT['boot'][r][TF[1]]['verdict']}"
    print(f"{r:17s} mean {v['mean']:5.1f} (sd {v['mean_sd_seeds']:.2f})  vflip {v['vflip']:5.1f}  rot90 {v['rot90']:5.1f}  FPR95 {v['fpr95']:5.1f}{ex}")
    print('     ', ' '.join(f"{s[:8]} {v['per_set'][s]:5.1f}" for s in OOD))
