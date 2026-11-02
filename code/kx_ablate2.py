# kx_ablate2.py -- ABLATIONS of the frozen orientation-completed kNN on the development split (fresh, seed 0) AND on the selection data
#   (devall vs sel-, seed 0), so that every design choice can be checked against the selection data.  Same rows as kx_ablate.py, except
#   that the orbit-bank row now uses R' = 4R and d re-estimated on the orbit bank (the bank is four times larger).
#   Fisher row: empirical lower-tail p-values calibrated on devall (selection: two-fold cross-fitting; fresh: all of devall).
#   v2: the orbit bank's d is estimated with single-precision cosines (lid32): with half precision the cosines between the four rotated
#   views of a Fashion-MNIST bank image round to 1, all 20 neighbour radii tie and the estimate is infinite.  The frozen d of the upright
#   bank is unchanged (it comes from the export); its single-precision value is recorded in _meta for reference only.
import os, sys, json, numpy as np, torch
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/upknn'); tag, gpu = sys.argv[1], int(sys.argv[2]); dev = torch.device(f'cuda:{gpu}')
CH = {'edmc10': ('L012', 2), 'l6svhn': ('L012', 5), 'dpceleba': ('L012', 5), 'edmc100': ('L012', 5), 'in64': ('L012', 1000), 'fmnist': ('L012', 1)}
h, R = CH[tag]; HS = {'L1': [1], 'L012': [0, 1, 2]}
def auroc(a, b):
    r = rankdata(np.concatenate([a, b])); return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))
ALL = [r[0] for r in json.load(open(f'runs/ok/sets_{tag}.json'))]; NAT = json.load(open('code/kx_nat.json'))[tag]
idn = [n for n in ALL if n.startswith('devall-')][0]; fid = 'fresh-' + idn[7:]; vf = fid + '__vflip'; SEL = [n for n in ALL if n.startswith('sel-')]
SETS = [idn, fid, vf] + ['fresh-' + o for o in NAT] + SEL
Z = np.load(f'runs/kx/export/{tag}_s0_fresh.npz'); d = float(Z['d']); T = float(Z['T'])
def load(name, l, bankdir=None, views=(0,)):
    X = np.load(f'{bankdir or "runs/ok/s0/" + tag}/k0/{name}__L{l}.npy', mmap_mode='r'); C = X.shape[2] // 10
    return torch.nn.functional.normalize(torch.from_numpy(np.ascontiguousarray(X[:, list(views), :2 * C])).to(dev).float(), dim=-1)
def space(name, bankdir=None, views=(0,)): return (torch.cat([load(name, l, bankdir, views) for l in HS[h]], -1) / np.sqrt(len(HS[h]))).half()
def lid(B, k=20, m=3000):
    idx = torch.from_numpy(np.random.RandomState(0).choice(len(B), min(m, len(B)), replace=False)).to(dev)
    c = (B[idx] @ B.T).float().topk(k + 1, dim=1).values[:, 1:].clamp(max=1 - 1e-6); r = torch.sqrt(2 * (1 - c))
    return float((1 / torch.log(r[:, -1:] / r[:, :-1]).mean(1)).mean())
def space32(name, bankdir=None, views=(0,)): return torch.cat([load(name, l, bankdir, views) for l in HS[h]], -1) / np.sqrt(len(HS[h]))
def lid32(B, k=20, m=3000, chunk=50000):
    idx = torch.from_numpy(np.random.RandomState(0).choice(len(B), min(m, len(B)), replace=False)).to(dev); Q = B[idx]; top = None
    for i in range(0, len(B), chunk):
        t = (Q @ B[i:i + chunk].T).topk(min(k + 1, len(B[i:i + chunk])), dim=1).values
        top = t if top is None else torch.cat([top, t], 1).topk(k + 1, dim=1).values
    c = top[:, 1:].clamp(max=1 - 1e-6); r = torch.sqrt(2 * (1 - c))
    return float((1 / torch.log(r[:, -1:] / r[:, :-1]).mean(1)).mean())
def content_vs(B, RR, dd):
    out = {}
    with torch.no_grad():
        for n in SETS:
            Q = space(n)[:, 0]; c = np.concatenate([(Q[i:i + 500] @ B.T).float().topk(RR, dim=1).values[:, -1].cpu().numpy() for i in range(0, len(Q), 500)]).astype(np.float64)
            out[n] = -dd * 0.5 * np.log(2 * (1 - np.minimum(c, 1 - 1e-9)))
    return out
Bu = space('bank')[:, 0]; C = content_vs(Bu, R, d)
B32 = space32('bank')[:, 0]; d32 = lid32(B32); del B32; torch.cuda.empty_cache()
B32 = space32('bank', f'runs/kx/rotb/s0/{tag}', (0, 1, 2, 3)); B32 = B32.reshape(-1, B32.shape[-1]); do = lid32(B32); del B32; torch.cuda.empty_cache()
Bo = space('bank', f'runs/kx/rotb/s0/{tag}', (0, 1, 2, 3)); Bo = Bo.reshape(-1, Bo.shape[-1]); do16 = lid(Bo); Corb = content_vs(Bo, 4 * R, do); del Bo; torch.cuda.empty_cache()
src = f'runs/kx/rothead/s0/{tag}_k0_gqR'; old = f'runs/kx/rothead/s0/{tag}_k0_gq'
def logsm(lg, T): z = lg / T; m = z.max(1, keepdims=True); return z[:, 0] - (m[:, 0] + np.log(np.exp(z - m).sum(1)))
P = {n: logsm(np.load(f'{src}/{n}.npz')['lg0_mlp'].astype(np.float64), T) for n in SETS}
P1 = {n: logsm(np.load(f'{src}/{n}.npz')['lg0_mlp'].astype(np.float64), 1.0) for n in SETS}
Pleak = {n: np.load(f'{old}/{n}.npz')['lp_mlp'][:, 0].astype(np.float64) for n in SETS}
P4 = {n: np.load(f'{src}/{n}.npz')['lp_mlp'].astype(np.float64).mean(1) for n in SETS}
def logp(ref, x): r = np.sort(ref); return np.log((1 + np.searchsorted(r, x, 'right')) / (len(r) + 1))
ROWS = {'ours': lambda n: C[n] + P[n], 'content only': lambda n: C[n], 'orientation only': lambda n: P[n], 'no temperature': lambda n: C[n] + P1[n],
        'orbit bank': lambda n: Corb[n] + P[n], 'head with rotated noise': lambda n: C[n] + Pleak[n], 'four views (rotated-noise test views)': lambda n: C[n] + P4[n]}
perm = np.random.RandomState(0).permutation(len(C[idn])); hh = len(perm) // 2; folds = [(perm[:hh], perm[hh:]), (perm[hh:], perm[:hh])]
res = {}
for k, f in ROWS.items():
    fr = {o: 100 * auroc(f(fid), f('fresh-' + o)) for o in NAT}
    res[k] = dict(nat=float(np.mean(list(fr.values()))), vflip=100 * auroc(f(fid), f(vf)), pairs=fr, sel=float(np.mean([100 * auroc(f(idn), f(s)) for s in SEL])))
fr = {o: 100 * auroc(logp(C[idn], C[fid]) + logp(P[idn], P[fid]), logp(C[idn], C['fresh-' + o]) + logp(P[idn], P['fresh-' + o])) for o in NAT}
FV = 100 * auroc(logp(C[idn], C[fid]) + logp(P[idn], P[fid]), logp(C[idn], C[vf]) + logp(P[idn], P[vf]))
sv = []
for cal, ev in folds:
    g = lambda n: logp(C[idn][cal], C[n]) + logp(P[idn][cal], P[n])
    sv.append(np.mean([100 * auroc(g(idn)[ev], g(s)) for s in SEL]))
res['Fisher instead of sum'] = dict(nat=float(np.mean(list(fr.values()))), vflip=FV, pairs=fr, sel=float(np.mean(sv)))
res['_meta'] = dict(d=d, d_orbit=do, R=R, R_orbit=4 * R, T=T, d_orbit_half=do16, d_upright_single=d32, version=2)
for k, v in res.items():
    if k != '_meta': print(f'{tag:8s} {k:40s} sel {v["sel"]:6.2f}  nat {v["nat"]:6.2f}  vflip {v["vflip"]:5.1f}', flush=True)
print(tag, res['_meta'], flush=True)
json.dump(res, open(f'runs/kx/ablate2_{tag}.json', 'w'), indent=1)
