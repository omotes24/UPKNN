# ok_prof.py -- PROTOCOL_OK.md extension 3 (2026-10-31 22:25 JST): the similarity profile to the bank in the kNN spaces (L1, L012;
#   sigma_1 and sigma_7 upright views): the r-th largest cosine for r in R and the mean of the r largest, for every image of the sets.
#   output: runs/ok/prof/s<seed>/<tag>__<set>.npz (P_<space>_k<k> [N, len(R)] float32 r-th values, M_<space>_k<k> [N, len(R)] means)
#   usage: python code/ok_prof.py --tag edmc10 --gpu 0 --seed 0 --prefix devall-,sel-
import os, json, time, argparse
import numpy as np
import torch
ROOT = os.path.expanduser('~/upknn')
RS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--gpu', type=int, default=0)
p.add_argument('--prefix', default='devall-,sel-'); p.add_argument('--seed', type=int, default=0)
a = p.parse_args()
dev = torch.device(f'cuda:{a.gpu}')
SETS = [r[0] for r in json.load(open(f'{ROOT}/runs/ok/sets_{a.tag}.json')) if r[0].startswith(tuple(a.prefix.split(',')))]
od = f'{ROOT}/runs/ok/prof/s{a.seed}'; os.makedirs(od, exist_ok=True)


def load(name, k, l):
    X = np.load(f'{ROOT}/runs/ok/s{a.seed}/{a.tag}/k{k}/{name}__L{l}.npy', mmap_mode='r'); C = X.shape[2] // 10
    return torch.nn.functional.normalize(torch.from_numpy(np.ascontiguousarray(X[:, 0, :2 * C])).to(dev).float(), dim=1)


t0 = time.time(); OUT = {n: {} for n in SETS}
for k in (0, 6):
    B = {l: load('bank', k, l) for l in range(3)}; nb = len(B[0]); rs = [r for r in RS if r <= nb]
    for n in SETS:
        Q = {l: load(n, k, l) for l in range(3)}
        for sp, hooks in (('L1', [1]), ('L012', [0, 1, 2])):
            Pv, Mv = [], []
            for i in range(0, len(Q[1]), 500):
                S = sum(Q[l][i:i + 500] @ B[l].T for l in hooks) / len(hooks)
                t = S.topk(max(rs), dim=1).values; del S
                Pv.append(torch.stack([t[:, r - 1] for r in rs], 1).cpu().numpy())
                cs = t.cumsum(1); Mv.append(torch.stack([cs[:, r - 1] / r for r in rs], 1).cpu().numpy())
            OUT[n][f'P_{sp}_k{k}'] = np.concatenate(Pv); OUT[n][f'M_{sp}_k{k}'] = np.concatenate(Mv)
    del B; torch.cuda.empty_cache()
for n in SETS: np.savez(f'{od}/{a.tag}__{n}.npz', R=np.array(RS), **OUT[n])
print('OK_PROF_DONE', a.tag, a.seed, f'{time.time() - t0:.0f}s', flush=True)
