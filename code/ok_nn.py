# ok_nn.py -- PROTOCOL_OK.md extension 2 (2026-10-31 22:20 JST): the 50 nearest bank images of every image in the kNN space of
#   PROTOCOL_CT (cosine of the L2-normalised global [mean, std] of hook L1 at sigma_1, upright +eps view, seed's noise) and in the
#   L012 space; written with their similarities, so that statistics of the image can be compared with those of its content neighbours.
#   output: runs/ok/nn/s<seed>/<tag>__<set>.npz (i_L1 [N, 50] int32, s_L1 [N, 50] float16, i_L012, s_L012)
#   usage: python code/ok_nn.py --tag edmc10 --gpu 0 --seed 0 --prefix devall-,sel-
import os, json, time, argparse
import numpy as np
import torch
ROOT = os.path.expanduser('~/upknn')
p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--gpu', type=int, default=0)
p.add_argument('--prefix', default='devall-,sel-'); p.add_argument('--seed', type=int, default=0)
a = p.parse_args()
dev = torch.device(f'cuda:{a.gpu}')
SETS = [r[0] for r in json.load(open(f'{ROOT}/runs/ok/sets_{a.tag}.json')) if r[0].startswith(tuple(a.prefix.split(',')))]
od = f'{ROOT}/runs/ok/nn/s{a.seed}'; os.makedirs(od, exist_ok=True)


def load(name, l):
    X = np.load(f'{ROOT}/runs/ok/s{a.seed}/{a.tag}/k0/{name}__L{l}.npy', mmap_mode='r'); C = X.shape[2] // 10
    return torch.nn.functional.normalize(torch.from_numpy(np.ascontiguousarray(X[:, 0, :2 * C])).to(dev).float(), dim=1)


t0 = time.time()
B = {l: load('bank', l) for l in range(3)}
for n in SETS:
    Q = {l: load(n, l) for l in range(3)}; out = {}
    for hname, hooks in (('L1', [1]), ('L012', [0, 1, 2])):
        I, V = [], []
        for i in range(0, len(Q[1]), 1000):
            S = sum(Q[l][i:i + 1000] @ B[l].T for l in hooks) / len(hooks); t = S.topk(50, dim=1)
            I.append(t.indices.cpu().numpy().astype(np.int32)); V.append(t.values.cpu().numpy().astype(np.float16)); del S
        out[f'i_{hname}'] = np.concatenate(I); out[f's_{hname}'] = np.concatenate(V)
    np.savez(f'{od}/{a.tag}__{n}.npz', **out)
print('OK_NN_DONE', a.tag, a.seed, len(SETS), f'{time.time() - t0:.0f}s', flush=True)
