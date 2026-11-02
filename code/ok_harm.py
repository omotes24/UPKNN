# ok_harm.py -- rotational-harmonic retrieval statistics from the features of ok_extract.py (GPU, read-only on the inputs).
#   Per view r (R_r(x + s_k eps), shared noise) the global pooled feature g_r = [channel mean, channel std] (2C) of hook l is
#   L2-normalised (n_r).  Harmonics of the orbit response under the rotation group C4 (pooled features need no re-indexing):
#     m0 = (n0+n1+n2+n3)/4 (invariant), m1 = (n0-n2)/2 (up/down dipole), m3 = (n1-n3)/2 (left/right dipole),
#     m2 = (n0-n1+n2-n3)/4 (0/180 vs 90/270), m02 = (n0+n2)/2 (invariant of the 2-view subgroup)
#   Representations (each block L2-normalised; a concatenation of B blocks is divided by sqrt(B) so that its cosine is the mean of
#   the block cosines with ONE bank image):
#     f0, m0, m1, m2, m3, H4 = [m0, m1, m2, m3], H2 = [m02, m1], H01 = [n0, m1], O4 = [n0, n1, n2, n3], O2 = [n0, n2]
#   H[N, 4 hook groups (L0, L1, L2, L012), R representations, 2 ranks (1st, 10th largest cosine over the bank)] float16
#   output: runs/ok/harm/s<seed>/<tag>/k<k>/<name>.npz (H, reps, idx)
#   usage: python code/ok_harm.py --tag edmc10 --seed 0 --k 0 --gpu 1 --prefix devall-,sel-   (--gpu -1: CPU, float32)
import os, json, time, argparse
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
dev = torch.device(f'cuda:{a.gpu}' if a.gpu >= 0 else 'cpu'); DT = torch.float16 if a.gpu >= 0 else torch.float32
if a.gpu < 0: torch.set_num_threads(int(os.environ.get('HARM_THREADS', '12')))
src = f'{ROOT}/runs/ok/s{a.seed}/{a.tag}/k{a.k}'
od = f'{ROOT}/runs/ok/harm/s{a.seed}/{a.tag}/k{a.k}'; os.makedirs(od, exist_ok=True)
SETS = [r[0] for r in json.load(open(f'{ROOT}/runs/ok/sets_{a.tag}.json')) if r[0].startswith(tuple(a.prefix.split(',')))]
SETS = [n for n in SETS if not os.path.exists(f'{od}/{n}.npz')]
REPS = ['f0', 'm0', 'm1', 'm2', 'm3', 'H4', 'H2', 'H01', 'O4', 'O2']
print(a.tag, a.seed, a.k, len(SETS), 'sets to do', flush=True)
nz = torch.nn.functional.normalize


def reps(X):
    """X [N, 4, 10C] float16 numpy -> dict rep -> L2-normalised half [N, D] on the GPU"""
    C = X.shape[2] // 10
    n = nz(torch.from_numpy(np.ascontiguousarray(X[:, :, :2 * C])).to(dev).float(), dim=2)        # [N, 4, 2C]
    m0 = n.mean(1); m1 = (n[:, 0] - n[:, 2]) / 2; m3 = (n[:, 1] - n[:, 3]) / 2
    m2 = (n[:, 0] - n[:, 1] + n[:, 2] - n[:, 3]) / 4; m02 = (n[:, 0] + n[:, 2]) / 2
    cat = lambda *b: torch.cat([nz(x, dim=1) for x in b], 1) / np.sqrt(len(b))
    out = {'f0': n[:, 0], 'm0': nz(m0, dim=1), 'm1': nz(m1, dim=1), 'm2': nz(m2, dim=1), 'm3': nz(m3, dim=1),
           'H4': cat(m0, m1, m2, m3), 'H2': cat(m02, m1), 'H01': cat(n[:, 0], m1), 'O4': cat(n[:, 0], n[:, 1], n[:, 2], n[:, 3]),
           'O2': cat(n[:, 0], n[:, 2])}
    return {k: v.to(DT) for k, v in out.items()}


t0 = time.time()
bank = {l: np.load(f'{src}/bank__L{l}.npy', mmap_mode='r') for l in range(3)}
B = {}
for l in range(3):                                                    # bank representations, built in chunks
    parts = []
    for c0 in range(0, len(bank[l]), 5000):
        parts.append(reps(bank[l][c0:c0 + 5000]))
    B[l] = {r: torch.cat([p_[r] for p_ in parts], 0) for r in REPS}
    del parts
print('bank reps', {l: tuple(B[l]['H4'].shape) for l in range(3)}, f'{time.time() - t0:.0f}s', flush=True)
with torch.no_grad():
    for n in SETS:
        Q = {l: np.load(f'{src}/{n}__L{l}.npy', mmap_mode='r') for l in range(3)}
        N = len(Q[0]); H = np.zeros((N, 4, len(REPS), 2), np.float16)
        for c0 in range(0, N, a.chunk):
            c1 = min(N, c0 + a.chunk)
            q = {l: reps(Q[l][c0:c1]) for l in range(3)}
            for ri, r in enumerate(REPS):
                acc = None
                for l in range(3):
                    S = (q[l][r] @ B[l][r].T).float()
                    top = S.topk(10, dim=1).values
                    H[c0:c1, l, ri, 0] = top[:, 0].cpu().numpy(); H[c0:c1, l, ri, 1] = top[:, 9].cpu().numpy()
                    acc = S if acc is None else acc.add_(S)
                top = (acc / 3).topk(10, dim=1).values
                H[c0:c1, 3, ri, 0] = top[:, 0].cpu().numpy(); H[c0:c1, 3, ri, 1] = top[:, 9].cpu().numpy()
                del acc, S
        np.savez(f'{od}/{n}.npz', H=H, reps=np.array(REPS), idx=np.load(f'{src}/{n}__idx.npy'))
        print(n, N, f'{time.time() - t0:.0f}s', flush=True)
print('OK_HARM_DONE', a.tag, a.seed, a.k, f'{time.time() - t0:.0f}s', flush=True)
