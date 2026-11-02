# e5_ana.py -- PROTOCOL_E5.md (analysis written before the results were read): Upright kNN and CFS on the full OpenOOD v1.5
#   ImageNet-1K sets at 64x64 as the bank grows.  Banks: the protocol's 10k (10 per class; tag in64, results of the frozen run) and
#   nested random 10k / 25k / 50k subsets of the HF training shard (tags in64h10 / in64h25 / in64h50).  Rows per bank: Ours with
#   R = N/10 (primary) and R = 1000, the kNN term (R = N/10), the orientation term, kNN (basic), CFS (diag / kNN heads); seeds 0-2.
#   Metrics: near / far AUROC and FPR95, csID AUROC, full spectrum (ID + csID positive, every image of the full sets weight 1).
#   usage: python e5_ana.py <e5 results dir> <ood1k results dir>          -> e5_ana.json
import os, sys, json
import numpy as np
SCR = os.environ.get('UPKNN_ANALYSIS', os.path.expanduser('~/upknn/analysis'))
HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else f'{SCR}/final/e5/res'; REF = sys.argv[2] if len(sys.argv) > 2 else f'{SCR}/final/ood1k/res'
sys.path.insert(0, f'{SCR}/eu_build/Euphonium/analysis'); sys.path.insert(0, f'{SCR}/final/ana')
from common import auroc, fpr95
ID = 'in64_imagenet_test'; NEAR = ['in64_ssb_hard', 'in64_ninco']; FAR = ['in64_inaturalist', 'in64_textures', 'in64_openimage_o']
CS = ['in64_imagenet_v2', 'in64_imagenet_c', 'in64_imagenet_r']; ALLS = [ID] + NEAR + FAR + CS; SEEDS = (0, 1, 2)


def rows_ref():
    R = {}
    for sd in SEEDS:
        U = np.load(f'{REF}/runs/kx/export/in64_s{sd}_ood1k.npz'); K = np.load(f'{REF}/runs/ood1k/knn_s{sd}.npz')
        for s in ALLS:
            n = f'ood1k-{s}'; assert (U[f'idx__{n}'] == K[f'idx__{n}']).all()
            for r, z, key in (('Ours (R=N/10)', U, 'score'), ('Ours (R=1000)', U, 'score'), ('kNN term', U, 'content'),
                              ('Orientation term', U, 'pose'), ('kNN basic', K, 'ct')):
                R.setdefault(r, {}).setdefault(sd, {})[s] = (np.asarray(z[f'{key}__{n}'], np.float64), np.asarray(U[f'idx__{n}']))
            C = np.load(f'{REF}/runs/cfs_l7/s{sd}/in64/{n}__scores.npz')
            R.setdefault('CFS', {}).setdefault(sd, {})[s] = (C['diag'].astype(np.float64), C['idx'])
            R.setdefault('CFS kNN', {}).setdefault(sd, {})[s] = (C['knn'].astype(np.float64), C['idx'])
    return R, 10000


def rows_bank(tag):
    R = {}; N = None
    for sd in SEEDS:
        p = f'{RES}/runs/e5/{tag}_s{sd}.npz'
        if not os.path.exists(p): continue
        E = np.load(p); N = int(E['N']); Rp = N // 10
        for s in ALLS:
            n = f'ood1k-{s}'; ix = np.asarray(E[f'idx__{n}']); pose = E[f'pose__{n}'].astype(np.float64)
            R.setdefault('Ours (R=N/10)', {}).setdefault(sd, {})[s] = (E[f'content_R{Rp}__{n}'] + pose, ix)
            R.setdefault('Ours (R=1000)', {}).setdefault(sd, {})[s] = (E[f'content_R1000__{n}'] + pose, ix)
            R.setdefault('kNN term', {}).setdefault(sd, {})[s] = (E[f'content_R{Rp}__{n}'].astype(np.float64), ix)
            R.setdefault('Orientation term', {}).setdefault(sd, {})[s] = (pose, ix)
            R.setdefault('kNN basic', {}).setdefault(sd, {})[s] = (E[f'ct__{n}'].astype(np.float64), ix)
        cd = f'{RES}/runs/cfs_l7/s{sd}/{tag}'
        if all(os.path.exists(f'{cd}/ood1k-{s}__scores.npz') for s in ALLS):
            for s in ALLS:
                C = np.load(f'{cd}/ood1k-{s}__scores.npz')
                R.setdefault('CFS', {}).setdefault(sd, {})[s] = (C['diag'].astype(np.float64), C['idx'])
                R.setdefault('CFS kNN', {}).setdefault(sd, {})[s] = (C['knn'].astype(np.float64), C['idx'])
    return R, N


def metrics(get):
    a = get(ID); o = {}
    for s in NEAR + FAR + CS: b = get(s); o[s] = dict(auroc=100 * auroc(a, b), fpr95=100 * fpr95(a, b))
    for g, L in (('near', NEAR), ('far', FAR)):
        o[g] = float(np.mean([o[s]['auroc'] for s in L])); o[g + '_fpr95'] = float(np.mean([o[s]['fpr95'] for s in L]))
    pos = np.concatenate([get(s) for s in [ID] + CS])
    for g, L in (('FS-near', NEAR), ('FS-far', FAR)): o[g] = float(np.mean([100 * auroc(pos, get(s)) for s in L]))
    return o


OUT = {}
for tag in ('in64', 'in64h10', 'in64h25', 'in64h50'):
    R, N = rows_ref() if tag == 'in64' else rows_bank(tag)
    if not R: print(tag, 'no results yet'); continue
    T = {'N': N, 'rows': {}}
    for r in R:
        per = [metrics(lambda s, sd=sd: R[r][sd][s][0]) for sd in sorted(R[r])]
        T['rows'][r] = {k: (float(np.mean([p[k] for p in per])) if not isinstance(per[0][k], dict) else {kk: float(np.mean([p[k][kk] for p in per])) for kk in per[0][k]}) for k in per[0]}
        T['rows'][r]['sd_near'] = float(np.std([p['near'] for p in per])); T['rows'][r]['sd_far'] = float(np.std([p['far'] for p in per])); T['rows'][r]['seeds'] = len(per)
    OUT[tag] = T
    print(f'== {tag} (N = {N})')
    for r, v in T['rows'].items():
        print(f"  {r:17s} near {v['near']:5.1f} (sd {v['sd_near']:.2f})  far {v['far']:5.1f} (sd {v['sd_far']:.2f})  FPR {v['near_fpr95']:5.1f}/{v['far_fpr95']:5.1f}  "
              f"V2 {v['in64_imagenet_v2']['auroc']:5.1f} C {v['in64_imagenet_c']['auroc']:5.1f} R {v['in64_imagenet_r']['auroc']:5.1f}  FS {v['FS-near']:5.1f}/{v['FS-far']:5.1f}")
json.dump(OUT, open(f'{HERE}/e5_ana.json', 'w'), indent=1)
