# ok_sims.py -- similarity statistics of PROTOCOL_OK.md from the features of ok_extract.py (GPU):
#   S[N, 4 hook groups (L0, L1, L2, L012), 3 feature definitions (g, q, gq), 4 query views r, 4 banks s, 2 ranks (1st, 10th)] float16
#   S = kk-th largest cosine over the bank B_s of phi(V_r(x)) (L012: mean over the three hooks of the per-hook cosine, then the rank)
#   output: runs/ok/sims/s<seed>/<tag>/k<k>/<name>.npz (S, e = the errors of the four views, idx)
#   usage: python code/ok_sims.py --tag edmc10 --seed 0 --k 0 --gpu 0 --prefix devall-,sel-
import os, sys, json, time, argparse
import numpy as np
import torch
ROOT = os.path.expanduser('~/upknn')
p = argparse.ArgumentParser()
p.add_argument('--tag', required=True)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--k', type=int, default=0)
p.add_argument('--gpu', type=int, default=0)
p.add_argument('--prefix', default='devall-,sel-')
p.add_argument('--chunk', type=int, default=1000)
a = p.parse_args()
dev = torch.device(f'cuda:{a.gpu}')
src = f'{ROOT}/runs/ok/s{a.seed}/{a.tag}/k{a.k}'
od = f'{ROOT}/runs/ok/sims/s{a.seed}/{a.tag}/k{a.k}'; os.makedirs(od, exist_ok=True)
SETS = [r[0] for r in json.load(open(f'{ROOT}/runs/ok/sets_{a.tag}.json')) if r[0].startswith(tuple(a.prefix.split(',')))]
SETS = [n for n in SETS if not os.path.exists(f'{od}/{n}.npz')]
print(a.tag, a.seed, a.k, len(SETS), 'sets to do', flush=True)


def phi(X, f):
    """X [N, 4, 10C] float16 (numpy) -> dict f -> L2-normalised torch half [N, 4, D] on the GPU"""
    C = X.shape[2] // 10; T = torch.from_numpy(np.ascontiguousarray(X)).to(dev).float()
    sl = {'g': slice(0, 2 * C), 'q': slice(2 * C, 10 * C), 'gq': slice(0, 10 * C)}[f]
    return torch.nn.functional.normalize(T[:, :, sl], dim=2).half()


t0 = time.time()
bank = {l: np.load(f'{src}/bank__L{l}.npy', mmap_mode='r') for l in range(3)}
Q = {n: {l: np.load(f'{src}/{n}__L{l}.npy', mmap_mode='r') for l in range(3)} for n in SETS}
OUT = {n: np.zeros((len(Q[n][0]), 4, 3, 4, 4, 2), np.float16) for n in SETS}
with torch.no_grad():
    for fi, f in enumerate(('g', 'q', 'gq')):
        B = {l: phi(bank[l], f) for l in range(3)}                       # [Nb, 4, D]
        for n in SETS:
            N = len(Q[n][0])
            for c0 in range(0, N, a.chunk):
                c1 = min(N, c0 + a.chunk)
                q = {l: phi(Q[n][l][c0:c1], f) for l in range(3)}
                for r in range(4):
                    for s in range(4):
                        acc = None
                        for l in range(3):
                            S = q[l][:, r] @ B[l][:, s].T                    # [c, Nb] half
                            S = S.float()
                            top = S.topk(10, dim=1).values
                            OUT[n][c0:c1, l, fi, r, s, 0] = top[:, 0].cpu().numpy(); OUT[n][c0:c1, l, fi, r, s, 1] = top[:, 9].cpu().numpy()
                            acc = S if acc is None else acc.add_(S)
                        top = (acc / 3).topk(10, dim=1).values
                        OUT[n][c0:c1, 3, fi, r, s, 0] = top[:, 0].cpu().numpy(); OUT[n][c0:c1, 3, fi, r, s, 1] = top[:, 9].cpu().numpy()
                        del acc, S
        del B; torch.cuda.empty_cache()
        print(f'{a.tag} k{a.k} phi {f} {time.time() - t0:.0f}s', flush=True)
for n in SETS:
    np.savez(f'{od}/{n}.npz', S=OUT[n], e=np.load(f'{src}/{n}__e.npy'), idx=np.load(f'{src}/{n}__idx.npy'))
print('OK_SIMS_DONE', a.tag, a.seed, a.k, f'{time.time() - t0:.0f}s', flush=True)
