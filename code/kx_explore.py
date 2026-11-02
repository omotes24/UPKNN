# kx_explore.py -- kNN extensions on the stored decoder features of runs/ok (ok_extract.py): joint similarity over several
#   components (noise levels k, hooks L, rotated views r, feature slices), the similarity profile (r-th largest cosine P_R and mean of
#   the R largest M_R), AUROC on the SELECTION data (devall-* vs sel-*) and on the development split (fresh-*).  final2-* is never read.
#   A space is "c1+c2+..." with components "L<l>.k<k>.r<r>.<f>" (f: g = [channel mean, std], q = 2x2-grid [mean, std], gq both,
#   m = channel mean only, s = channel std only) or drift "L<l>.d<k1>-<k0>.r<r>.<f>" (difference of the raw pooled features of two noise
#   levels, then normalised).  The joint similarity is the mean of the component cosines (each component L2-normalised).
#   usage: python code/kx_explore.py --tag edmc10 --gpu 0 --spaces A,B,... --out runs/kx/explore_<name>.json [--seed 0] [--fresh 1]
import os, json, time, argparse
import numpy as np
import torch
ROOT = os.path.expanduser('~/upknn')
RS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
p = argparse.ArgumentParser()
p.add_argument('--tag', required=True); p.add_argument('--gpu', type=int, default=0); p.add_argument('--seed', type=int, default=0)
p.add_argument('--spaces', required=True); p.add_argument('--out', required=True); p.add_argument('--fresh', type=int, default=1)
a = p.parse_args()
dev = torch.device(f'cuda:{a.gpu}')
ALL = [r[0] for r in json.load(open(f'{ROOT}/runs/ok/sets_{a.tag}.json'))]
assert not any(n.startswith('final2-') for n in ALL if n.startswith(('devall-', 'sel-', 'fresh-')))
SETS = [n for n in ALL if n.startswith(('devall-', 'sel-'))] + ([n for n in ALL if n.startswith('fresh-')] if a.fresh else [])
MM = {}


def raw(name, k, l):
    key = (name, k, l)
    if key not in MM: MM[key] = np.load(f'{ROOT}/runs/ok/s{a.seed}/{a.tag}/k{k}/{name}__L{l}.npy', mmap_mode='r')
    return MM[key]


def sl(f, C):
    return {'g': np.r_[0:2 * C], 'q': np.r_[2 * C:10 * C], 'gq': np.r_[0:10 * C], 'm': np.r_[0:C], 's': np.r_[C:2 * C]}[f]


def comp(name, c):
    """component c of the set name -> L2-normalised float32 torch [N, D] on the GPU"""
    L, kk, r, f = c.split('.'); l = int(L[1]); r = int(r[1])
    if kk.startswith('k'):
        X = raw(name, int(kk[1:]), l); C = X.shape[2] // 10
        T = torch.from_numpy(np.ascontiguousarray(X[:, r][:, sl(f, C)])).to(dev).float()
    elif kk.startswith('o'):                                       # orbit difference o<k>-<r2>: view r minus view r2 at noise level k
        k, r2 = [int(v) for v in kk[1:].split('-')]
        X = raw(name, k, l); C = X.shape[2] // 10
        T = torch.from_numpy(np.ascontiguousarray(X[:, r][:, sl(f, C)])).to(dev).float() - torch.from_numpy(np.ascontiguousarray(X[:, r2][:, sl(f, C)])).to(dev).float()
    else:                                                          # drift d<k1>-<k0>
        k1, k0 = [int(v) for v in kk[1:].split('-')]
        X1, X0 = raw(name, k1, l), raw(name, k0, l); C = X1.shape[2] // 10
        T = torch.from_numpy(np.ascontiguousarray(X1[:, r][:, sl(f, C)])).to(dev).float() - torch.from_numpy(np.ascontiguousarray(X0[:, r][:, sl(f, C)])).to(dev).float()
    return torch.nn.functional.normalize(T, dim=1)


def space(name, sp):
    cs = sp.split('+'); return torch.cat([comp(name, c) for c in cs], 1) / np.sqrt(len(cs))


def auroc(x_id, x_ood):
    a_, b_ = np.asarray(x_id, np.float64), np.asarray(x_ood, np.float64)
    from scipy.stats import rankdata
    rk = rankdata(np.concatenate([a_, b_])); return float((rk[:len(a_)].sum() - len(a_) * (len(a_) + 1) / 2) / (len(a_) * len(b_)))


OUT = json.load(open(f'{ROOT}/{a.out}')) if os.path.exists(f'{ROOT}/{a.out}') else {}
t0 = time.time()
with torch.no_grad():
    for sp in a.spaces.split(','):
        if sp in OUT: continue
        B = space('bank', sp).half(); nb = len(B); rs = [r for r in RS if r <= nb]
        prof = {}
        for n in SETS:
            Q = space(n, sp).half(); Pv, Mv = [], []
            for i in range(0, len(Q), 1000):
                t = (Q[i:i + 1000] @ B.T).float().topk(max(rs), dim=1).values
                Pv.append(torch.stack([t[:, r - 1] for r in rs], 1).cpu().numpy())
                cs = t.cumsum(1); Mv.append(torch.stack([cs[:, r - 1] / r for r in rs], 1).cpu().numpy())
            prof[n] = (np.concatenate(Pv), np.concatenate(Mv))
        del B; torch.cuda.empty_cache()
        idn = [n for n in SETS if n.startswith('devall-')][0]; sel = [n for n in SETS if n.startswith('sel-')]
        fid = [n for n in SETS if n.startswith('fresh-') and n.endswith(idn[7:])]
        res = {'sel': {}, 'fresh': {}}
        for st, si in (('P', 0), ('M', 1)):
            for j, r in enumerate(rs):
                res['sel'][f'{st}{r}'] = {s: 100 * auroc(prof[idn][si][:, j], prof[s][si][:, j]) for s in sel}
                if fid:
                    res['fresh'][f'{st}{r}'] = {s: 100 * auroc(prof[fid[0]][si][:, j], prof[s][si][:, j]) for s in SETS if s.startswith('fresh-') and s != fid[0]}
        OUT[sp] = res
        best = max(res['sel'], key=lambda k: np.mean(list(res['sel'][k].values())))
        print(f'{a.tag} {sp:60s} sel best {best:6s} {np.mean(list(res["sel"][best].values())):6.2f}  ({time.time() - t0:.0f}s)', flush=True)
        os.makedirs(os.path.dirname(f'{ROOT}/{a.out}'), exist_ok=True); json.dump(OUT, open(f'{ROOT}/{a.out}', 'w'))
print('KX_EXPLORE_DONE', a.tag, f'{time.time() - t0:.0f}s', flush=True)
