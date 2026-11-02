# kx_rh2scores.py -- converts the rotation-head outputs (kx_rothead.py) into score files of runs/kx/scores (kx_ana2.py format):
#   RH<k>.<feat>.p0<head> = log p(0 | x) (1 pass), RH<k>.<feat>.rot4<head> = sum_r log p(r | R_r x) (4 passes), and the kNN profile in
#   the MLP hidden space of view 0 (RH<k>.<feat>.H0; cosine, P_R / M_R).  usage: python code/kx_rh2scores.py --tag edmc10 --gpu 0
import os, json, argparse, numpy as np, torch
ROOT = os.path.expanduser('~/upknn'); RS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--gpu', type=int, default=0)
p.add_argument('--k', type=int, default=0); p.add_argument('--feat', default='gq'); p.add_argument('--seed', type=int, default=0); a = p.parse_args()
dev = torch.device(f'cuda:{a.gpu}')
src = f'{ROOT}/runs/kx/rothead/s{a.seed}/{a.tag}_k{a.k}_{a.feat}'; od = f'{ROOT}/runs/kx/scores/s{a.seed}/{a.tag}'; os.makedirs(od, exist_ok=True)
sets = sorted(f[:-4] for f in os.listdir(src) if f.endswith('.npz'))
pre = f'RH{a.k}.{a.feat}'
for hn in ('lin', 'mlp'):
    for kind in ('p0', 'rot4'):
        out = {}
        for n in sets:
            Z = np.load(f'{src}/{n}.npz'); lp = Z[f'lp_{hn}'].astype(np.float64)
            v = lp[:, 0] if kind == 'p0' else lp.sum(1)
            out[n] = np.stack([v[:, None], v[:, None]], 1).astype(np.float32); out['idx__' + n] = Z['idx']
        np.savez(f'{od}/{pre}.{kind}{hn}.npz', RS=np.array([1]), **out)
B = torch.nn.functional.normalize(torch.from_numpy(np.load(f'{src}/bank__H0.npy')).to(dev).float(), dim=1).half(); rs = [r for r in RS if r <= len(B)]
out = {}
with torch.no_grad():
    for n in sets:
        Z = np.load(f'{src}/{n}.npz'); Q = torch.nn.functional.normalize(torch.from_numpy(Z['H0']).to(dev).float(), dim=1).half(); V = []
        for i in range(0, len(Q), 1000):
            t = (Q[i:i + 1000] @ B.T).float().topk(max(rs), dim=1).values; cs = t.cumsum(1)
            V.append(torch.stack([torch.stack([t[:, r - 1] for r in rs], 1), torch.stack([cs[:, r - 1] / r for r in rs], 1)], 1).cpu().numpy())
        out[n] = np.concatenate(V).astype(np.float32); out['idx__' + n] = Z['idx']
np.savez(f'{od}/{pre}.H0.npz', RS=np.array(rs), **out)
print('KX_RH2SCORES_DONE', a.tag, flush=True)
