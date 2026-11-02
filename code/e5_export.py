# e5_export.py <tag> <gpu> <seed> -- PROTOCOL_E5.md: Upright kNN on the ood1k- sets with the bank of <tag>: content terms for
#   R = 1000 and R = N / 10, the orientation term of the head trained on this bank, and kNN (basic: L1 [mean, std], 10th neighbour).
#   As kx_export_x.py otherwise (L0+L1+L2, [mean, std] of view 0, cosine; d by Levina-Bickel on 3,000 bank points, finite estimates).
#   -> runs/e5/<tag>_s<seed>.npz {content_R<R>__<set>, pose__<set>, ct__<set>, idx__<set>, d, T, N}
import os, sys, json
import numpy as np, torch, torch.nn.functional as F
ROOT = os.path.expanduser('~/upknn'); tag, gpu, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); dev = torch.device(f'cuda:{gpu}')
SETS = [r[0] for r in json.load(open(f'{ROOT}/runs/ok/sets_{tag}.json')) if r[0].startswith('ood1k-')]
D = f'{ROOT}/runs/ok/s{seed}/{tag}/k0'
def blk(name, l):
    X = np.load(f'{D}/{name}__L{l}.npy', mmap_mode='r'); C = X.shape[2] // 10
    return F.normalize(torch.from_numpy(np.ascontiguousarray(X[:, 0, :2 * C])).to(dev).float(), dim=1)
def space(name): return (torch.cat([blk(name, l) for l in (0, 1, 2)], 1) / np.sqrt(3)).half()
B = space('bank'); N = len(B); RS = sorted({1000, N // 10}); B1 = blk('bank', 1).half()
idx = torch.from_numpy(np.random.RandomState(0).choice(N, min(3000, N), replace=False)).to(dev)
with torch.no_grad():
    c = (B[idx] @ B.T).float().topk(21, dim=1).values[:, 1:].clamp(max=1 - 1e-6); r = torch.sqrt(2 * (1 - c))
    dd = 1 / torch.log(r[:, -1:] / r[:, :-1]).mean(1); fin = torch.isfinite(dd); d = float(dd[fin].mean())
src = f'{ROOT}/runs/kx/rothead/s{seed}/{tag}_k0_gqR'; T = json.load(open(f'{src}/temp.json'))['mlp']
def logsm(lg, T): z = lg / T; m = z.max(1, keepdims=True); return z[:, 0] - (m[:, 0] + np.log(np.exp(z - m).sum(1)))
out = {'d': d, 'T': T, 'N': N, 'RS': np.array(RS)}
with torch.no_grad():
    for n in SETS:
        Q = space(n); Q1 = blk(n, 1).half(); cs = {R: [] for R in RS}; ct = []
        for i in range(0, len(Q), 1000):
            t = (Q[i:i + 1000] @ B.T).float().topk(max(RS), dim=1).values
            for R in RS: cs[R].append(t[:, R - 1].cpu().numpy())
            ct.append((Q1[i:i + 1000] @ B1.T).float().topk(10, dim=1).values[:, -1].cpu().numpy())
        for R in RS:
            cR = np.concatenate(cs[R]).astype(np.float64); out[f'content_R{R}__{n}'] = -d * 0.5 * np.log(2 * (1 - np.minimum(cR, 1 - 1e-9)))
        Z = np.load(f'{src}/{n}.npz'); out[f'pose__{n}'] = logsm(Z['lg0_mlp'].astype(np.float64), T)
        out[f'ct__{n}'] = np.concatenate(ct).astype(np.float64); out[f'idx__{n}'] = np.load(f'{D}/{n}__idx.npy')
        assert (out[f'idx__{n}'] == Z['idx']).all()
os.makedirs(f'{ROOT}/runs/e5', exist_ok=True); np.savez(f'{ROOT}/runs/e5/{tag}_s{seed}.npz', **out)
print('E5_EXPORT_DONE', tag, seed, 'N', N, 'RS', RS, 'd', round(d, 2), 'T', round(T, 3), len(SETS), 'sets', flush=True)
