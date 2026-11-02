# kx_ablate.py -- ABLATIONS on the development split (fresh, noise seed 0) of the frozen orientation-completed kNN: the full score,
#   each term alone, no temperature, Fisher combination instead of the sum, the orbit bank for the content term, the head trained with
#   the rotated noise (leak), four views for the orientation term.  Writes runs/kx/ablate_<tag>.json (natural mean, vflip, per pair).
import os, sys, json, numpy as np, torch
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/upknn'); tag, gpu = sys.argv[1], int(sys.argv[2]); dev = torch.device(f'cuda:{gpu}')
CH = {'edmc10': ('L012', 2), 'l6svhn': ('L012', 5), 'dpceleba': ('L012', 5), 'edmc100': ('L012', 5), 'in64': ('L012', 1000), 'fmnist': ('L012', 1)}
h, R = CH[tag]; HS = {'L1': [1], 'L012': [0, 1, 2]}
def auroc(a, b):
    r = rankdata(np.concatenate([a, b])); return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))
ALL = [r[0] for r in json.load(open(f'runs/ok/sets_{tag}.json'))]; NAT = json.load(open('code/kx_nat.json'))[tag]
idn = [n for n in ALL if n.startswith('devall-')][0]; fid = 'fresh-' + idn[7:]; vf = fid + '__vflip'
SETS = [idn, fid, vf] + ['fresh-' + o for o in NAT]
Z = np.load(f'runs/kx/export/{tag}_s0_fresh.npz'); d = float(Z['d']); T = float(Z['T'])
def load(name, l, bankdir=None, views=(0,)):
    X = np.load(f'{bankdir or "runs/ok/s0/" + tag}/k0/{name}__L{l}.npy', mmap_mode='r'); C = X.shape[2] // 10
    return torch.nn.functional.normalize(torch.from_numpy(np.ascontiguousarray(X[:, list(views), :2 * C])).to(dev).float(), dim=-1)
def space(name, bankdir=None, views=(0,)): return (torch.cat([load(name, l, bankdir, views) for l in HS[h]], -1) / np.sqrt(len(HS[h]))).half()
def content_vs(B):
    out = {}
    with torch.no_grad():
        for n in SETS:
            Q = space(n)[:, 0]; c = np.concatenate([(Q[i:i + 1000] @ B.T).float().topk(R, dim=1).values[:, -1].cpu().numpy() for i in range(0, len(Q), 1000)]).astype(np.float64)
            out[n] = -d * 0.5 * np.log(2 * (1 - np.minimum(c, 1 - 1e-9)))
    return out
Bu = space('bank')[:, 0]; C = content_vs(Bu)
Bo = space('bank', f'runs/kx/rotb/s0/{tag}', (0, 1, 2, 3)); Bo = Bo.reshape(-1, Bo.shape[-1]); Corb = content_vs(Bo); del Bo
src = f'runs/kx/rothead/s0/{tag}_k0_gqR'; old = f'runs/kx/rothead/s0/{tag}_k0_gq'
def logsm(lg, T): z = lg / T; m = z.max(1, keepdims=True); return z[:, 0] - (m[:, 0] + np.log(np.exp(z - m).sum(1)))
P = {n: logsm(np.load(f'{src}/{n}.npz')['lg0_mlp'].astype(np.float64), T) for n in SETS}
P1 = {n: logsm(np.load(f'{src}/{n}.npz')['lg0_mlp'].astype(np.float64), 1.0) for n in SETS}
Pleak = {n: np.load(f'{old}/{n}.npz')['lp_mlp'][:, 0].astype(np.float64) for n in SETS}
P4 = {n: np.load(f'{src}/{n}.npz')['lp_mlp'].astype(np.float64).mean(1) for n in SETS}
def logp(ref, x): r = np.sort(ref); return np.log((1 + np.searchsorted(r, x, 'right')) / (len(r) + 1))
FI = {n: logp(C[idn], C[n]) + logp(P[idn], P[n]) for n in SETS}
ROWS = {'ours': {n: C[n] + P[n] for n in SETS}, 'content only': C, 'orientation only': P, 'no temperature': {n: C[n] + P1[n] for n in SETS},
        'Fisher instead of sum': FI, 'orbit bank': {n: Corb[n] + P[n] for n in SETS}, 'head with rotated noise': {n: C[n] + Pleak[n] for n in SETS},
        'four views (rotated-noise test views)': {n: C[n] + P4[n] for n in SETS}}
res = {}
for k, S in ROWS.items():
    fr = {o: 100 * auroc(S[fid], S['fresh-' + o]) for o in NAT}; res[k] = dict(nat=float(np.mean(list(fr.values()))), vflip=100 * auroc(S[fid], S[vf]), pairs=fr)
    print(f'{tag:8s} {k:40s} nat {res[k]["nat"]:6.2f} vflip {res[k]["vflip"]:5.1f}', flush=True)
json.dump(res, open(f'runs/kx/ablate_{tag}.json', 'w'), indent=1)
