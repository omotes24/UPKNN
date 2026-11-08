# vl_protocol.py -- writes runs/vl/PROTOCOL_VL.json: the per-ID choices of Vertical Limit, made WITHOUT any evaluation OOD image.
#   Inputs: vl_eval.py results of noise seed 0 (runs/vl/eval/<tag>_s0_k<k>_full.json/_scores.npz), of which only the ID images of the
#   development split and their vertical flips (mirror test) and the selection data (devall ID vs sel-) are used.
#   1. level l*  = argmax over single decoder levels of the mirror test M(l, sigma_0, R = 1)
#   2. R*        = argmax over R of the selection-data AUROC of the kNN term at (l*, sigma_0)          (as Upright kNN chose R)
#   3. a0        = 1 if adding the expert [that kNN term + log q(up|x)] to the Upright-kNN score raises the selection-data AUROC, else 0
#   4. sigma*    = argmax over the extracted noise levels (sigma_0..sigma_4) of M(l*, sigma, R*); if sigma* != sigma_0: w* = max(0, 2 M* - 1), else w* = 0;
#                  w* = 0 also when the bootstrap 95% interval of M* (2000 resamples of the ID images and of their flips) does not exclude 50
#   score = s_U + a0 * [c(l*, sigma_0, R*) + log q] + w* * [c(l*, sigma*, R*) + log q]   (every content term paired with the pose term)
#   usage: python code/vl_protocol.py [tags]   (prints the choices and writes / updates the json)
import os, json, numpy as np
from scipy.stats import rankdata
import sys
TAGS = sys.argv[1:] or ['edmc10', 'l6svhn', 'dpceleba', 'edmc100', 'in64', 'fmnist']
RUP = {'edmc10': 2, 'l6svhn': 5, 'dpceleba': 5, 'edmc100': 5, 'in64': 1000, 'fmnist': 1}           # Upright kNN (kx_export.py CH)


def auroc(x, y):
    r = rankdata(np.concatenate([x, y])); return 100 * float((r[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))


P = json.load(open('runs/vl/PROTOCOL_VL.json')) if os.path.exists('runs/vl/PROTOCOL_VL.json') else {}
for tag in TAGS:
    J = {k: json.load(open(f'runs/vl/eval/{tag}_s0_k{k}_full.json')) for k in range(5) if os.path.exists(f'runs/vl/eval/{tag}_s0_k{k}_full.json')}
    Z0 = np.load(f'runs/vl/eval/{tag}_s0_k0_full_scores.npz')
    levels = [s for s in J[0]['content'] if s in ('L0', 'L1', 'L2', 'L3')]
    M0 = {s: J[0]['rows'][f'content {s} R1']['vflip'] for s in levels}
    lv = max(levels, key=M0.get)
    Rs = sorted({int(r.split(' R')[1]) for r in J[0]['rows'] if r.startswith(f'content {lv} R')})
    R = max(Rs, key=lambda r: J[0]['rows'][f'content {lv} R{r}']['sel'])
    names = sorted({f.split('__', 1)[1] for f in Z0.files if f.startswith('pose__')})
    idn = [n for n in names if n.startswith('devall-')][0]; sel = [n for n in names if n.startswith('sel-')]
    U = {n: Z0[f'content_L012_R{RUP[tag]}__{n}'] + Z0[f'pose__{n}'] for n in names}
    T0 = {n: Z0[f'content_{lv}_R{R}__{n}'] + Z0[f'pose__{n}'] for n in names}   # an Upright-kNN expert at (l*, sigma_0, R*): content + pose
    selU = np.mean([auroc(U[idn], U[s]) for s in sel]); selT = np.mean([auroc(U[idn] + T0[idn], U[s] + T0[s]) for s in sel])
    a0 = 1 if selT > selU else 0
    Mk = {k: J[k]['rows'][f'content {lv} R{R}']['vflip'] for k in J if f'content {lv} R{R}' in J[k]['rows']}
    ks = max(Mk, key=Mk.get); ws = max(0.0, 2 * Mk[ks] / 100 - 1) if ks != 0 else 0.0
    ci = [None, None]
    if ks != 0:                                                       # is there vertical information at sigma* at all? (bootstrap CI of M*)
        Zk = np.load(f'runs/vl/eval/{tag}_s0_k{ks}_full_scores.npz'); fid = 'fresh-' + idn[7:]
        a_, b_ = Zk[f'content_{lv}_R{R}__{fid}'], Zk[f'content_{lv}_R{R}__{fid}__vflip']; rng = np.random.RandomState(20261031)
        bs = [auroc(a_[rng.randint(0, len(a_), len(a_))], b_[rng.randint(0, len(b_), len(b_))]) for _ in range(2000)]
        ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        if ci[0] <= 50: ws = 0.0
    P[tag] = {'level': int(lv[1:]), 'R': int(R), 'a0': a0, 'kstar': (int(ks) if ks != 0 else None), 'wstar': round(ws, 4), 'mirror_star_ci': ci,
              'mirror_levels': {s: round(v, 2) for s, v in M0.items()}, 'mirror_sigma': {f'k{k}': round(v, 2) for k, v in Mk.items()},
              'sel_upright': round(float(selU), 3), 'sel_with_term': round(float(selT), 3), 'R_upright': RUP[tag]}
    print(tag, P[tag], flush=True)
json.dump({t: P[t] for t in ['edmc10', 'l6svhn', 'dpceleba', 'edmc100', 'in64', 'fmnist'] if t in P}, open('runs/vl/PROTOCOL_VL.json', 'w'), indent=1)
print('PROTOCOL_VL_WRITTEN')
