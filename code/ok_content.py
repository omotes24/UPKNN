# ok_content.py -- PROTOCOL_OK.md extension 1 (2026-10-31 22:00 JST, selection data only): variants of the content statistic on the
#   features of ok_extract.py (upright view, global [mean, std]): raw cosine (= PROTOCOL_CT's kNN for L1, k0), centring, PCA whitening
#   (fit on the bank; top 64 / 256 / all components), concatenation of the two noise levels, local-density ratio (LOF-like).
#   Hook groups L1 and L012 (mean of the per-hook cosines).  Per-image scores (higher = more ID) of the selection sets.
#   output: runs/ok/content/s<seed>/<tag>__<set>.npz (one array per variant)
#   usage: python code/ok_content.py --tag edmc10 --gpu 0 [--prefix devall-,sel-]
import os, sys, json, time, argparse
import numpy as np
import torch
ROOT = os.path.expanduser('~/upknn')
p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--gpu', type=int, default=0)
p.add_argument('--prefix', default='devall-,sel-'); p.add_argument('--seed', type=int, default=0)
a = p.parse_args()
dev = torch.device(f'cuda:{a.gpu}')
SETS = [r[0] for r in json.load(open(f'{ROOT}/runs/ok/sets_{a.tag}.json')) if r[0].startswith(tuple(a.prefix.split(',')))]
od = f'{ROOT}/runs/ok/content/s{a.seed}'; os.makedirs(od, exist_ok=True)


def load(name, k, l):
    X = np.load(f'{ROOT}/runs/ok/s{a.seed}/{a.tag}/k{k}/{name}__L{l}.npy', mmap_mode='r'); C = X.shape[2] // 10
    return torch.from_numpy(np.ascontiguousarray(X[:, 0, :2 * C])).to(dev).float()


def nrm(X): return torch.nn.functional.normalize(X, dim=1)


def feat(name, k, h):
    if k == 'c': return torch.cat([nrm(load(name, 0, h)), nrm(load(name, 6, h))], 1)
    return load(name, k, h)


def transform(Bk, var):
    """returns f(X) for the variant, fitted on the bank features Bk"""
    if var == 'raw': return lambda X: nrm(X)
    mu = Bk.mean(0, keepdim=True)
    if var == 'cent': return lambda X: nrm(X - mu)
    Bc = Bk - mu; C = (Bc.T @ Bc) / len(Bc); ev, U = torch.linalg.eigh(C.double()); ev, U = ev.flip(0).float(), U.flip(1).float()
    d = min({'w64': 64, 'w256': 256, 'wall': len(ev)}[var], len(ev)); Wm = U[:, :d] / torch.sqrt(ev[:d] + 1e-3 * ev[0])
    return lambda X: nrm((X - mu) @ Wm)


def topk(fQ, fB, chunk=1000):
    """fQ, fB: lists over hooks; mean of the per-hook cosines; top-10 values and indices"""
    V, I = [], []
    for i in range(0, len(fQ[0]), chunk):
        S = sum(q[i:i + chunk] @ b.T for q, b in zip(fQ, fB)) / len(fQ)
        t = S.topk(10, dim=1); V.append(t.values); I.append(t.indices); del S
    return torch.cat(V), torch.cat(I)


t0 = time.time(); RES = {n: {} for n in SETS}
for hname, hooks in (('L1', [1]), ('L012', [0, 1, 2])):
    for k in (0, 6, 'c'):
        Bk = {h: feat('bank', k, h) for h in hooks}
        for var in ('raw', 'cent', 'w64', 'w256', 'wall'):
            tf = {h: transform(Bk[h], var) for h in hooks}; fB = [tf[h](Bk[h]) for h in hooks]
            for n in SETS:
                fQ = [tf[h](feat(n, k, h)) for h in hooks]
                v, i = topk(fQ, fB)
                RES[n][f'{hname}_k{k}_{var}_kk10'] = v[:, 9].cpu().numpy(); RES[n][f'{hname}_k{k}_{var}_kk1'] = v[:, 0].cpu().numpy()
                if var == 'raw' and k == 0: RES[n][f'_nn_{hname}'] = i.cpu().numpy()
                del fQ
            del fB; torch.cuda.empty_cache()
        if k == 0:                                   # LOF-like ratio: s10(x) / mean of s10(b) over the 10 nearest bank images b
            fB = [nrm(Bk[h]) for h in hooks]; nb = len(fB[0]); sb = np.zeros(nb, np.float32)
            for i in range(0, nb, 1000):
                S = sum(b[i:i + 1000] @ bb.T for b, bb in zip(fB, fB)) / len(fB)
                S[torch.arange(S.shape[0]), torch.arange(i, i + S.shape[0])] = -2
                sb[i:i + 1000] = S.topk(10, dim=1).values[:, 9].cpu().numpy(); del S
            for n in SETS:
                nn = RES[n].pop(f'_nn_{hname}'); s = RES[n][f'{hname}_k0_raw_kk10']
                RES[n][f'{hname}_k0_lofratio_kk10'] = s / sb[nn].mean(1); RES[n][f'{hname}_k0_lofdiff_kk10'] = s - sb[nn].mean(1)
            del fB
        del Bk; torch.cuda.empty_cache()
        print(a.tag, hname, k, f'{time.time() - t0:.0f}s', flush=True)
for n in SETS: np.savez(f'{od}/{a.tag}__{n}.npz', **RES[n])
print('OK_CONTENT_DONE', a.tag, f'{time.time() - t0:.0f}s', flush=True)
