# ok_devnn2.py -- attempt 4: appearance neighbours among the ID development images (devall) for query sets: cosine of the upright
#   global pooled feature at s1 (hook L1; ok_extract.py, shared noise of the seed), top-K dev images.  For devall itself the image is
#   excluded (leave-one-out).  output: runs/ok/devnn2/s<seed>/<tag>__<set>.npz (c [N, K] float16, i [N, K] int16 = rows of devall, idx)
#   usage: python code/ok_devnn2.py --tag edmc10 --seed 0 --prefix fresh-,final2- [--K 300] [--gpu -1]
import os, json, time, argparse
import numpy as np
import torch
ROOT = os.path.expanduser('~/upknn')
p = argparse.ArgumentParser()
p.add_argument('--tag', required=True); p.add_argument('--seed', type=int, default=0)
p.add_argument('--prefix', default='fresh-,final2-'); p.add_argument('--K', type=int, default=300); p.add_argument('--gpu', type=int, default=-1)
a = p.parse_args()
if a.gpu < 0: torch.set_num_threads(int(os.environ.get('HARM_THREADS', '8')))
S0 = f'{ROOT}/runs/ok/s{a.seed}/{a.tag}/k0'; od = f'{ROOT}/runs/ok/devnn2/s{a.seed}'; os.makedirs(od, exist_ok=True)
rows = json.load(open(f'{ROOT}/runs/ok/sets_{a.tag}.json'))
DEV = [r[0] for r in rows if r[0].startswith('devall-')][0]
Q = [DEV] + [r[0] for r in rows if r[0].startswith(tuple(a.prefix.split(',')))]
nz = torch.nn.functional.normalize
def up(name):
    X = np.load(f'{S0}/{name}__L1.npy', mmap_mode='r'); C = X.shape[2] // 10
    return nz(torch.from_numpy(np.ascontiguousarray(X[:, 0, :2 * C])).float(), dim=1)
D = up(DEV); t0 = time.time()
for n in Q:
    X = up(n); S = X @ D.T
    if n == DEV: S.fill_diagonal_(-2.0)
    v = S.topk(a.K, dim=1)
    np.savez(f'{od}/{a.tag}__{n}.npz', c=v.values.numpy().astype(np.float16), i=v.indices.numpy().astype(np.int16), idx=np.load(f'{S0}/{n}__idx.npy'))
print('OK_DEVNN2_DONE', a.tag, a.seed, len(Q), f'{time.time() - t0:.0f}s', flush=True)
