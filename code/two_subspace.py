# two_subspace.py -- POST-HOC ANALYSIS (no change to the method; development split 'fresh' and the bank, noise seed 0): where the
#   orientation information of the frozen features lies relative to the directions that dominate nearest-neighbour distances.
#   Per ID tier, from the stored features (runs/kx/rotb: the four rotated bank views; runs/ok: the sets):
#   (1) kNN space of Upright kNN (L0+L1+L2, [channel mean, std], unit length per block): distance between an image and its own
#       rotation (90 / 180 degrees) relative to its distance to its R-th nearest other bank image;
#   (2) head space (gq features of the three blocks, standardised on the bank as by kx_rothead2.py): the 3 linear discriminant
#       directions between the four rotations (LDA on 4,000 bank images x 4 views) and the share of the total variance they carry,
#       against the top-3 principal directions and against 3 random directions (3/D);
#   (3) kNN (R = 10, a random 10,000-image subset of the bank) restricted to the LDA subspace and to its orthogonal complement (in the
#       head space), and in the full head space: AUROC of ID vs upside-down ID images and vs every natural OOD set of the development split.
#   usage: python code/two_subspace.py <tag> [device]      -> runs/kx/two_subspace_<tag>.json
import os, sys, json, numpy as np, torch
ROOT = os.path.expanduser('~/upknn'); torch.set_num_threads(16); tag = sys.argv[1]; dev = torch.device(sys.argv[2] if len(sys.argv) > 2 else 'cpu')
CH = {'edmc10': 2, 'l6svhn': 5, 'dpceleba': 5, 'edmc100': 5, 'in64': 1000, 'fmnist': 1}; R = CH[tag]
SETS = [r for r in json.load(open(f'{ROOT}/runs/ok/sets_{tag}.json')) if r[0].startswith('fresh-')]
IDN = [r[0] for r in SETS if r[3] == 'none'][0]; VF = [r[0] for r in SETS if r[3] == 'vflip'][0]
OODS = [r[0] for r in SETS if r[3] == 'none' and r[0] != IDN and 'synth' not in r[0]]
rs = np.random.RandomState(0)
def bank(l):
    return np.load(f'{ROOT}/runs/kx/rotb/s0/{tag}/k0/bank__L{l}.npy', mmap_mode='r')        # [N, 4, 10C] float16
B = [bank(l) for l in range(3)]; N = B[0].shape[0]; Cs = [b.shape[2] // 10 for b in B]
sub = np.sort(rs.choice(N, min(4000, N), replace=False))
def gspace(X):        # X: list of three [n, 10C] arrays -> unit-length [mean, std] per block, concatenated / sqrt(3)
    return torch.cat([torch.nn.functional.normalize(torch.from_numpy(np.asarray(x[:, :2 * c], np.float32)), dim=1) for x, c in zip(X, Cs)], 1) / np.sqrt(3)
out = {'tag': tag, 'R': R, 'N': int(N)}
# (1) rotation displacement vs neighbour distance in the kNN space
G0 = gspace([b[:, 0] for b in B]).to(dev)                                                   # all bank images, view 0
for v, name in ((1, 'rot90'), (2, 'rot180')):
    Gv = gspace([b[sub, v] for b in B]).to(dev)
    d_rot = torch.sqrt(torch.clamp(2 * (1 - (G0[sub] * Gv).sum(1)), min=0))
    sims = G0[sub] @ G0.T; sims[torch.arange(len(sub)), torch.from_numpy(sub).to(dev)] = -2       # exclude the image itself
    cR = sims.topk(R, dim=1).values[:, -1]; d_nn = torch.sqrt(torch.clamp(2 * (1 - cR), min=0))
    out[f'ratio_{name}'] = dict(median=float((d_rot / d_nn).median()), q25=float((d_rot / d_nn).quantile(0.25)), q75=float((d_rot / d_nn).quantile(0.75)),
                                frac_rot_closer_than_nn=float((d_rot < d_nn).float().mean()))
# (2) LDA of the four rotations in the standardised gq head space
Xh = np.concatenate([np.asarray(b[sub], np.float32) for b in B], 2)                       # [n, 4, D]
mu = Xh.reshape(-1, Xh.shape[2]).mean(0); sd = Xh.reshape(-1, Xh.shape[2]).std(0) + 1e-3
Z = torch.from_numpy((Xh - mu) / sd).double().to(dev); n, _, D = Z.shape; del Xh
Zf = Z.reshape(-1, D); m = Zf.mean(0); St = (Zf - m).T @ (Zf - m) / Zf.shape[0]
mc = Z.mean(0)                                                                               # [4, D] rotation means
Sb = sum(torch.outer(mc[r] - m, mc[r] - m) for r in range(4)) / 4
Sw = St - Sb + 1e-3 * torch.eye(D, dtype=torch.float64, device=dev)
ev, U = torch.linalg.eigh(Sw); W = U @ torch.diag(ev.rsqrt()) @ U.T                          # Sw^(-1/2)
e2, V = torch.linalg.eigh(W @ Sb @ W); L = W @ V[:, -3:]                                      # top-3 discriminant directions
L = torch.linalg.qr(L).Q                                                                     # orthonormal basis of the LDA subspace
tot = torch.trace(St); share_lda = float(torch.trace(L.T @ St @ L) / tot)
pe = torch.linalg.eigvalsh(St); share_pca = float(pe[-3:].sum() / tot)
out['lda'] = dict(D=int(D), share_of_variance_lda3=share_lda, share_pca3=share_pca, share_random3=3 / D,
                  between_over_total=float(torch.trace(Sb) / tot))
# (3) kNN (R = 10) in the LDA subspace vs in its complement, development split
def setz(name):
    X = [np.load(f'{ROOT}/runs/ok/s0/{tag}/k0/{name}__L{l}.npy', mmap_mode='r') for l in range(3)]
    x = np.concatenate([np.asarray(a[:, 0], np.float32) for a in X], 1)
    return torch.from_numpy((x - mu) / sd).float().to(dev)
bsub = np.sort(np.random.RandomState(1).choice(N, min(10000, N), replace=False))
Zb = torch.from_numpy((np.concatenate([np.asarray(b[bsub, 0], np.float32) for b in B], 1) - mu) / sd).float().to(dev)
Lf = L.float(); del Z, Zf, St, Sb, Sw, U, W, V
def proj(x, which): return x if which == 'full' else (x @ Lf if which == 'lda' else x - (x @ Lf) @ Lf.T)
def knn_score(q, bk, k=10):
    qn, bn = torch.nn.functional.normalize(q, dim=1), torch.nn.functional.normalize(bk, dim=1)
    return torch.cat([(qn[i:i + 2000] @ bn.T).topk(k, dim=1).values[:, -1] for i in range(0, len(qn), 2000)]).cpu().numpy()
from scipy.stats import rankdata
def auroc(a, b): r = rankdata(np.concatenate([a, b])); return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))
res = {}
for which in ('full', 'lda', 'complement'):
    bk = proj(Zb, which); si = knn_score(proj(setz(IDN), which), bk)
    res[which] = {s.replace('fresh-', ''): 100 * auroc(si, knn_score(proj(setz(s), which), bk)) for s in [VF] + OODS}
out['knn_subspace'] = res
json.dump(out, open(f'{ROOT}/runs/kx/two_subspace_{tag}.json', 'w'), indent=1)
print(json.dumps(out)[:1500], flush=True)
