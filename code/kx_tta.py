# kx_tta.py -- DEVELOPMENT: pose from four rotated views (mean of log p(r | R_r x), stored test views; their noise IS rotated, an
#   approximation) vs the one-view pose, each added to the content term of kx_ofk2.py (seed 0)
import os, sys, json, numpy as np
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/upknn')
def auroc(a, b):
    r = rankdata(np.concatenate([a, b])); return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))
NATS = json.load(open('code/kx_nat.json'))
for tag in ['edmc10', 'l6svhn', 'dpceleba', 'edmc100', 'in64', 'fmnist']:
    Z = np.load(f'runs/kx/ofk2/{tag}_s0_scores.npz'); ALL = [r[0] for r in json.load(open(f'runs/ok/sets_{tag}.json'))]
    idn = [n for n in ALL if n.startswith('devall-')][0]; sel = [n for n in ALL if n.startswith('sel-')]; fid = 'fresh-' + idn[7:]; NAT = NATS[tag]
    src = f'runs/kx/rothead/s0/{tag}_k0_gqR'
    lp = {n: np.load(f'{src}/{n}.npz')['lp_mlp'].astype(np.float64) for n in [idn] + sel + [fid] + ['fresh-' + o for o in NAT] + [fid + '__vflip']}
    C = {n: Z[f'content__{n}'] for n in lp}
    for name, P in (('pose1', {n: Z[f'pose_gq.mlp__{n}'] for n in lp}), ('pose4', {n: lp[n].mean(1) for n in lp}), ('pose4sum', {n: lp[n].sum(1) for n in lp})):
        S = {n: C[n] + P[n] for n in lp}
        s_ = np.mean([100 * auroc(S[idn], S[s]) for s in sel]); fr = [100 * auroc(S[fid], S['fresh-' + o]) for o in NAT]
        print(f'{tag:8s} content+{name:8s} sel {s_:6.2f} fresh {np.mean(fr):6.2f} vflip {100 * auroc(S[fid], S[fid + "__vflip"]):5.1f}  ' + ' '.join(f'{v:5.1f}' for v in fr), flush=True)
