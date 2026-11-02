# vl_fscore_x.py -- vl_fscore.py (frozen) with one change: the split may also be 'ood1k' (the OpenOOD ImageNet-1K benchmark sets of
#   prep_ood1k.py, scored after the protocol was frozen and evaluated).  The scores are computed exactly as in vl_fscore.py.
# vl_fscore.py -- Vertical Limit scores of one ID, noise seed and split, from the frozen protocol runs/vl/PROTOCOL_VL.json:
#   score = [Upright kNN: -d_t log r_{R_t} (L0+L1+L2, sigma_0) + log q(up | x)]                    (runs/kx/export, unchanged)
#         + a0 * [(-d log r_R*) at the vertical level l* and sigma_0 + log q(up | x)]                (the deepest decoder level)
#         + w* * [(-d log r_R*) at l* and the vertical-limit noise level sigma* + log q(up | x)]    (if sigma* != sigma_0)
#   d = Levina-Bickel dimension of the bank in the same space (k = 20, 3000 bank points, RandomState(0)), cosine kNN, rank R*.
#   output: runs/vl/export/<tag>_s<seed>_<split>.npz  (score__, upright__, deep0__, deepk__, idx__ per set; d0, dk)
#   usage: python code/vl_fscore.py <tag> <gpu> <seed> <split: fresh|final2>
import os, sys, json, hashlib
import numpy as np, torch
ROOT = os.path.expanduser('~/upknn')
tag, gpu, seed, split = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
assert split in ('fresh', 'final2', 'ood1k')
PF = f'{ROOT}/runs/vl/PROTOCOL_VL.json'; P = json.load(open(PF))[tag]
dev = torch.device(f'cuda:{gpu}')
U = np.load(f'{ROOT}/runs/kx/export/{tag}_s{seed}_{split}.npz')
names = [f[7:] for f in U.files if f.startswith('score__')]
lv, R = P['level'], P['R']


def load(k, name):
    X = np.load(f'{ROOT}/runs/vl/featf/s{seed}/{tag}/k{k}/{name}__L{lv}.npy')
    return torch.nn.functional.normalize(torch.from_numpy(X).to(dev).float(), dim=1).half(), np.load(f'{ROOT}/runs/vl/featf/s{seed}/{tag}/k{k}/{name}__idx.npy')


@torch.no_grad()
def term(k):
    B, _ = load(k, 'bank')
    idx = torch.from_numpy(np.random.RandomState(0).choice(len(B), min(3000, len(B)), replace=False)).to(dev)
    c = torch.cat([(B[idx[i:i + 500]] @ B.T).float().topk(21, dim=1).values[:, 1:] for i in range(0, len(idx), 500)]).clamp(max=1 - 1e-6)
    r = torch.sqrt(2 * (1 - c)); d = float((1 / torch.log(r[:, -1:] / r[:, :-1]).mean(1)).mean())
    out = {}
    for n in names:
        Q, qi = load(k, n); assert (qi == U[f'idx__{n}']).all(), n
        cR = torch.cat([(Q[i:i + 500] @ B.T).float().topk(R, dim=1).values[:, -1] for i in range(0, len(Q), 500)]).cpu().numpy().astype(np.float64)
        out[n] = -d * 0.5 * np.log(2 * (1 - np.minimum(cR, 1 - 1e-9)))
    return out, d


up = {n: U[f'score__{n}'].astype(np.float64) for n in names}
res = {'d0': np.nan, 'dk': np.nan}
T0 = {n: np.zeros_like(up[n]) for n in names}; Tk = {n: np.zeros_like(up[n]) for n in names}
if P['a0'] > 0: T0, res['d0'] = term(0)
if P['kstar'] is not None and P['wstar'] > 0: Tk, res['dk'] = term(P['kstar'])
for n in names:
    res[f'upright__{n}'] = up[n]; res[f'deep0__{n}'] = T0[n]; res[f'deepk__{n}'] = Tk[n]; res[f'idx__{n}'] = U[f'idx__{n}']
    pz = U[f'pose__{n}'].astype(np.float64)                          # every content expert is paired with the upright posterior
    res[f'score__{n}'] = up[n] + P['a0'] * (T0[n] + pz) + P['wstar'] * (Tk[n] + pz)
os.makedirs(f'{ROOT}/runs/vl/export', exist_ok=True)
np.savez(f'{ROOT}/runs/vl/export/{tag}_s{seed}_{split}.npz', protocol_md5=hashlib.md5(open(PF, 'rb').read()).hexdigest(), **res)
print('VL_FSCORE_DONE', tag, seed, split, 'd0', round(float(res['d0']), 2), 'dk', round(float(res['dk']), 2), len(names), 'sets', flush=True)
