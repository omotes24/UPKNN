# kx_export.py -- per-image scores of the orientation-completed kNN for one ID and noise seed, with the content choice FIXED from the
#   seed-0 selection (CH below) and the leak-free MLP head on [L0, L1, L2] x gq (temperature from held-out bank images).
#   score = -d * log r_R + log P_T(up | x);  also writes the content and pose terms separately (ablations).
#   usage: python code/kx_export.py <tag> <gpu> <seed> <prefixes, e.g. fresh- or final2->     -> runs/kx/export/<tag>_s<seed>_<prefix>.npz
import os, sys, json, numpy as np, torch
ROOT = os.path.expanduser('~/upknn')
CH = {'edmc10': ('L012', 2), 'l6svhn': ('L012', 5), 'dpceleba': ('L012', 5), 'edmc100': ('L012', 5), 'in64': ('L012', 1000), 'fmnist': ('L012', 1)}
HS = {'L1': [1], 'L012': [0, 1, 2]}
tag, gpu, seed, prefix = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
dev = torch.device(f'cuda:{gpu}')
SETS = [r[0] for r in json.load(open(f'{ROOT}/runs/ok/sets_{tag}.json')) if r[0].startswith(tuple(prefix.split(',')))]
h, R = CH[tag]
def load(name, l):
    X = np.load(f'{ROOT}/runs/ok/s{seed}/{tag}/k0/{name}__L{l}.npy', mmap_mode='r'); C = X.shape[2] // 10
    return torch.nn.functional.normalize(torch.from_numpy(np.ascontiguousarray(X[:, 0, :2 * C])).to(dev).float(), dim=1)
def space(name): return (torch.cat([load(name, l) for l in HS[h]], 1) / np.sqrt(len(HS[h]))).half()
B = space('bank')
idx = torch.from_numpy(np.random.RandomState(0).choice(len(B), min(3000, len(B)), replace=False)).to(dev)
with torch.no_grad():
    c = (B[idx] @ B.T).float().topk(21, dim=1).values[:, 1:].clamp(max=1 - 1e-6); r = torch.sqrt(2 * (1 - c))
    d = float((1 / torch.log(r[:, -1:] / r[:, :-1]).mean(1)).mean())
src = f'{ROOT}/runs/kx/rothead/s{seed}/{tag}_k0_gqR'; T = json.load(open(f'{src}/temp.json'))['mlp']
def logsm(lg, T): z = lg / T; m = z.max(1, keepdims=True); return z[:, 0] - (m[:, 0] + np.log(np.exp(z - m).sum(1)))
out = {'d': d, 'T': T, 'R': R}
with torch.no_grad():
    for n in SETS:
        Q = space(n); cR = np.concatenate([(Q[i:i + 1000] @ B.T).float().topk(R, dim=1).values[:, -1].cpu().numpy() for i in range(0, len(Q), 1000)]).astype(np.float64)
        content = -d * 0.5 * np.log(2 * (1 - np.minimum(cR, 1 - 1e-9)))
        Z = np.load(f'{src}/{n}.npz') if os.path.exists(f'{src}/{n}.npz') else None
        if Z is None: raise SystemExit(f'missing head output for {n} (run kx_rothead2.py with the prefix)')
        pose = logsm(Z['lg0_mlp'].astype(np.float64), T)
        out[f'content__{n}'] = content; out[f'pose__{n}'] = pose; out[f'score__{n}'] = content + pose
        out[f'idx__{n}'] = np.load(f'{ROOT}/runs/ok/s{seed}/{tag}/k0/{n}__idx.npy'); assert (out[f'idx__{n}'] == Z['idx']).all()
os.makedirs(f'{ROOT}/runs/kx/export', exist_ok=True)
np.savez(f'{ROOT}/runs/kx/export/{tag}_s{seed}_{prefix.replace(",", "_").replace("-", "")}.npz', **out)
print('KX_EXPORT_DONE', tag, seed, prefix, 'd', round(d, 2), 'T', round(T, 3), len(SETS), 'sets', flush=True)
