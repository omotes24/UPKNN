# ok_jkde.py -- joint appearance-behaviour density (PROTOCOL_OK.md 2026-11-01 family "joint density of what the model sees and
#   how it behaves").  For an image x:
#     a(x) = L2-normalised global pooled decoder feature of the upright view at s1 (hook L1, or L012 = mean of the three per-hook
#            cosines), B(x) = behaviour vector from the denoising errors of ok_extract.py (shared noise):
#            l0 = log e(x) at s1, l6 = log e(x) at s7, a_r = log e(R_r x)/e(x) at s1 (r = 1, 2, 3)
#     score(x) = log sum_{b in top-1000 appearance neighbours} exp(kappa (cos(a(x), a(b)) - 1) - |W (B(x) - B(b) - delta)|^2 / (2 h^2))
#   over the training bank; delta, W (whitening) from the ID calibration images: the residual B(x) - B(nn(x)) to the nearest bank
#   image (removes the train/test offset of the errors).  h = inf: appearance only.
#   --mode sel  : calibration = one half of devall (2 folds, permutation RandomState(0) as ana_ok), queries = devall other half + sel-*
#   --mode final: calibration = all of devall, queries = the sets of --prefix (e.g. final2-)
#   output: runs/ok/jkde/sel/s<seed>/<tag>.npz or runs/ok/jkde/final_<prefix>/s<seed>/<tag>.npz : cands (strings), and per query set S__<set> [N, n_cands(, 2 folds)] float32
#   usage: python code/ok_jkde.py --tag edmc10 --mode sel --gpu -1
import os, json, time, argparse, itertools
import numpy as np
import torch
ROOT = os.path.expanduser('~/upknn')
p = argparse.ArgumentParser()
p.add_argument('--tag', required=True)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--mode', default='sel')
p.add_argument('--prefix', default='final2-')
p.add_argument('--gpu', type=int, default=-1)
p.add_argument('--top', type=int, default=1000)
a = p.parse_args()
dev = torch.device(f'cuda:{a.gpu}' if a.gpu >= 0 else 'cpu')
if a.gpu < 0: torch.set_num_threads(int(os.environ.get('HARM_THREADS', '12')))
S0 = f'{ROOT}/runs/ok/s{a.seed}/{a.tag}'
od = f'{ROOT}/runs/ok/jkde/' + (a.mode if a.mode == 'sel' else 'final_' + a.prefix.rstrip('-')) + f'/s{a.seed}'; os.makedirs(od, exist_ok=True)
rows = json.load(open(f'{ROOT}/runs/ok/sets_{a.tag}.json'))
IDSET = [r[0] for r in rows if r[0].startswith('devall-')][0]
QSETS = [r[0] for r in rows if r[0].startswith('sel-')] if a.mode == 'sel' else [r[0] for r in rows if r[0].startswith(a.prefix)]
KAPPA = [30, 100, 300]; H = [1.0, 2.0, 4.0]
BEH = {'a2l0': ['l0', 'a2'], 'a2l2': ['l0', 'l6', 'a2'], 'o3l0': ['l0', 'a1', 'a2', 'a3'], 'o3l2': ['l0', 'l6', 'a1', 'a2', 'a3']}
CANDS = [(hk, k, 'none', 0.0) for hk in ('L1', 'L012') for k in KAPPA] + \
        [(hk, k, b, h) for hk in ('L1', 'L012') for k in KAPPA for b in BEH for h in H]
nz = torch.nn.functional.normalize
t0 = time.time()


def upright(name, l):
    X = np.load(f'{S0}/k0/{name}__L{l}.npy', mmap_mode='r'); C = X.shape[2] // 10
    return nz(torch.from_numpy(np.ascontiguousarray(X[:, 0, :2 * C])).float(), dim=1)


def behav(name):
    e0 = np.load(f'{S0}/k0/{name}__e.npy'); e6 = np.load(f'{S0}/k6/{name}__e.npy')
    l0 = np.log(e0[:, 0])
    return {'l0': l0, 'l6': np.log(e6[:, 0]), 'a1': np.log(e0[:, 1]) - l0, 'a2': np.log(e0[:, 2]) - l0, 'a3': np.log(e0[:, 3]) - l0}


B_bank = {l: upright('bank', l).to(dev) for l in range(3)}
Bb = behav('bank')
print(a.tag, 'bank', B_bank[1].shape, f'{time.time() - t0:.0f}s', flush=True)


def topk(name):
    """dict hook -> (cos [N, top], idx [N, top]) for L1 and L012"""
    Q = {l: upright(name, l) for l in range(3)}; N = len(Q[1]); out = {}
    for hk in ('L1', 'L012'):
        C, I = [], []
        for c0 in range(0, N, 500):
            if hk == 'L1': S = Q[1][c0:c0 + 500].to(dev) @ B_bank[1].T
            else: S = sum(Q[l][c0:c0 + 500].to(dev) @ B_bank[l].T for l in range(3)) / 3
            v = S.topk(a.top, dim=1); C.append(v.values.cpu()); I.append(v.indices.cpu())
        out[hk] = (torch.cat(C).numpy().astype(np.float64), torch.cat(I).numpy())
    return out


def scores(T, Bq, cal_T, cal_B):
    """T: topk dict of the query set; Bq: behaviour dict; cal_*: same for the calibration images -> [N, n_cands]"""
    out = np.zeros((len(Bq['l0']), len(CANDS)))
    calib = {}
    for bn, keys in BEH.items():
        for hk in ('L1', 'L012'):
            Xc = np.stack([cal_B[k] for k in keys], 1); Nc = np.stack([Bb[k] for k in keys], 1)[cal_T[hk][1][:, 0]]
            r = Xc - Nc; delta = r.mean(0); C = np.cov(r.T).reshape(len(keys), len(keys))
            calib[(bn, hk)] = (delta, np.linalg.cholesky(np.linalg.inv(C + 1e-9 * np.eye(len(keys)))))
    D2 = {}
    for bn, keys in BEH.items():
        for hk in ('L1', 'L012'):
            delta, Wm = calib[(bn, hk)]
            Xq = np.stack([Bq[k] for k in keys], 1); Nb = np.stack([Bb[k] for k in keys], 1)[T[hk][1]]
            D2[(bn, hk)] = (((Xq[:, None, :] - Nb - delta) @ Wm) ** 2).sum(2)
    for j, (hk, kappa, bn, h) in enumerate(CANDS):
        w = kappa * (T[hk][0] - 1)
        if bn != 'none': w = w - 0.5 * D2[(bn, hk)] / h ** 2
        m = w.max(1, keepdims=True); out[:, j] = m[:, 0] + np.log(np.exp(w - m).sum(1))
    return out


T_id = topk(IDSET); B_id = behav(IDSET)
res = {'cands': np.array(['|'.join(map(str, c)) for c in CANDS])}
if a.mode == 'sel':
    n = len(B_id['l0']); perm = np.random.RandomState(0).permutation(n); folds = [(perm[:n // 2], perm[n // 2:]), (perm[n // 2:], perm[:n // 2])]
    sub = lambda T, Bv, ix: ({hk: (T[hk][0][ix], T[hk][1][ix]) for hk in T}, {k: v[ix] for k, v in Bv.items()})
    for f, (cal, ev) in enumerate(folds):
        cT, cB = sub(T_id, B_id, cal); eT, eB = sub(T_id, B_id, ev)
        res[f'S__{IDSET}__f{f}'] = scores(eT, eB, cT, cB).astype(np.float32); res[f'ev__f{f}'] = ev
    for s in QSETS:
        Ts, Bs = topk(s), behav(s)
        for f, (cal, ev) in enumerate(folds):
            cT, cB = sub(T_id, B_id, cal); res[f'S__{s}__f{f}'] = scores(Ts, Bs, cT, cB).astype(np.float32)
        print(s, f'{time.time() - t0:.0f}s', flush=True)
else:
    for s in QSETS:
        Ts, Bs = topk(s), behav(s)
        res[f'S__{s}'] = scores(Ts, Bs, T_id, B_id).astype(np.float32)
        res[f'idx__{s}'] = np.load(f'{S0}/k0/{s}__idx.npy')
        print(s, f'{time.time() - t0:.0f}s', flush=True)
np.savez(f'{od}/{a.tag}.npz', **res)
print('OK_JKDE_DONE', a.tag, a.mode, f'{time.time() - t0:.0f}s', flush=True)
