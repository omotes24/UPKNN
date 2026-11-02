# cfs_score.py -- the ID-only heads of CFS on the descriptors of cfs_extract.py (official cfs/methods/cfs.py):
#   diag (main paper): per slot the bank mean and biased variance (floor 1e-6), slot score = mean_j (z_j - mu_j)^2 / v_j, final score =
#     mean over the two slots;
#   knn (Appendix E.6, head_k = 10, head_standardize): z-score the concatenated descriptor with the bank mean / std (floor 1e-6),
#     Euclidean distance, score = mean distance to the 10 nearest bank descriptors.
#   Both are OOD-high in the paper; stored ID-high (negated) as runs/cfs/s<seed>/<tag>/<name>__scores.npz {diag, knn, idx}.
#   usage: python code/cfs_score.py --tag edmc10 --gpu 0 --seed 0 [--run runs/cfs_l7]
import os, sys, json, glob, argparse
import numpy as np
import torch
ROOT = os.path.expanduser('~/upknn')
p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--gpu', type=int, default=0); p.add_argument('--seed', type=int, default=0)
p.add_argument('--run', default='runs/cfs')
a = p.parse_args(); dev = torch.device(f'cuda:{a.gpu}'); EPS = 1e-6
od = f'{ROOT}/{a.run}/s{a.seed}/{a.tag}'; meta = json.load(open(f'{od}/meta.json'))
D1 = 2 * meta['shapes'][meta['hooks'][0]][1]                       # encoder slot width (2 C)
B = torch.from_numpy(np.load(f'{od}/bank__z.npy')).double().to(dev)
slots = [(0, D1), (D1, B.shape[1])]
stat = [(B[:, s:e].mean(0), torch.clamp(B[:, s:e].var(0, unbiased=False), min=EPS)) for s, e in slots]
mu, sd = B.mean(0), torch.clamp(B.std(0, unbiased=False), min=EPS); Bs = ((B - mu) / sd).float()
@torch.no_grad()
def score(Z):
    Z = Z.double().to(dev)
    diag = torch.stack([((Z[:, s:e] - m) ** 2 / v).mean(1) for (s, e), (m, v) in zip(slots, stat)], 1).mean(1)
    Zs = ((Z - mu) / sd).float(); kn = []
    for i in range(0, len(Zs), 1000):
        d = torch.cdist(Zs[i:i + 1000], Bs); kn.append(d.topk(10, dim=1, largest=False).values.mean(1))
    return (-diag).cpu().numpy(), (-torch.cat(kn)).double().cpu().numpy()
for f in sorted(glob.glob(f'{od}/*__z.npy')):
    name = os.path.basename(f)[:-len('__z.npy')]
    if name == 'bank': continue
    dg, kn = score(torch.from_numpy(np.load(f)))
    np.savez(f'{od}/{name}__scores.npz', diag=dg, knn=kn, idx=np.load(f'{od}/{name}__idx.npy'))
print('CFS_SCORE_DONE', a.tag, a.seed, flush=True)
