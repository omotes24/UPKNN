# two_subspace2.py -- POST-HOC ANALYSIS, robustness of two_subspace.py (no change to the method).  Per ID tier, noise seed and split
#   (final2 = the evaluation images, fresh = the development images), from the stored features (runs/kx/rotb/s<seed>: the four rotated
#   bank views; runs/ok/s<seed>: the sets):
#   (1) kNN space of Upright kNN (L0+L1+L2, [channel mean, std], unit length per block, / sqrt 3): distance between a bank image and its
#       own 90 / 180 degree rotation divided by its distance to its R-th nearest other bank image (R of the frozen protocol);
#   (2) head space (the gq features of the three blocks standardised on the fitting images): the share of the total variance carried by
#       the k = 1, 2, 3 linear discriminant directions of the four rotations, by the top-k principal directions and by k random
#       directions (20 draws); LDA fitted on 4,000 bank images x 4 views with ridge 1e-3 (as two_subspace.py) and, as variants, on
#       10,000 images and / or with ridge 1e-1;
#   (3) kNN in a 1-3 dimensional projection of the head space (bank: the 10,000-image subset of two_subspace.py; score: minus the
#       Euclidean distance to the 10th nearest bank image; for k = 3 also the cosine version of two_subspace.py): AUROC of the ID test
#       images against the upside-down ID images and against every natural OOD set of the split, for the LDA directions (every variant),
#       the top-3 principal directions and 3 random directions (20 draws, mean and range).
#   usage: python code/two_subspace2.py <tag> <seed> <split> [device] [all]  -> runs/kx/two_subspace2/<tag>_s<seed>_<split>.json
#          (the LDA variants only with 'all'; otherwise the setting of two_subspace.py alone)
import os, sys, json, time, numpy as np, torch
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/upknn'); torch.set_num_threads(12)
tag, seed, split = sys.argv[1], int(sys.argv[2]), sys.argv[3]; dev = torch.device(sys.argv[4] if len(sys.argv) > 4 else 'cpu')
assert split in ('final2', 'fresh')
CH = {'edmc10': 2, 'l6svhn': 5, 'dpceleba': 5, 'edmc100': 5, 'in64': 1000, 'fmnist': 1}; R = CH[tag]
SETS = [r for r in json.load(open(f'{ROOT}/runs/ok/sets_{tag}.json')) if r[0].startswith(f'{split}-')]
IDN = [r[0] for r in SETS if r[3] == 'none'][0]; VF = [r[0] for r in SETS if r[3] == 'vflip'][0]
OODS = [r[0] for r in SETS if r[3] == 'none' and r[0] != IDN and 'synth' not in r[0]]
t0 = time.time()
def bank(l): return np.load(f'{ROOT}/runs/kx/rotb/s{seed}/{tag}/k0/bank__L{l}.npy', mmap_mode='r')      # [N, 4, 10C] float16
B = [bank(l) for l in range(3)]; N = B[0].shape[0]; Cs = [b.shape[2] // 10 for b in B]
sub4 = np.sort(np.random.RandomState(0).choice(N, min(4000, N), replace=False))                         # as two_subspace.py
sub10 = np.sort(np.random.RandomState(2).choice(N, min(10000, N), replace=False))
bsub = np.sort(np.random.RandomState(1).choice(N, min(10000, N), replace=False))                        # kNN bank, as two_subspace.py
def gspace(X):
    return torch.cat([torch.nn.functional.normalize(torch.from_numpy(np.asarray(x[:, :2 * c], np.float32)), dim=1) for x, c in zip(X, Cs)], 1) / np.sqrt(3)
out = {'tag': tag, 'seed': seed, 'split': split, 'R': R, 'N': int(N), 'id': IDN, 'vflip': VF, 'ood': OODS}
# (1) rotation displacement / R-th neighbour distance in the kNN space
G0 = gspace([b[:, 0] for b in B]).to(dev)
for v, name in ((1, 'rot90'), (2, 'rot180')):
    Gv = gspace([b[sub4, v] for b in B]).to(dev)
    d_rot = torch.sqrt(torch.clamp(2 * (1 - (G0[sub4] * Gv).sum(1)), min=0))
    sims = G0[sub4] @ G0.T; sims[torch.arange(len(sub4)), torch.from_numpy(sub4).to(dev)] = -2
    d_nn = torch.sqrt(torch.clamp(2 * (1 - sims.topk(R, dim=1).values[:, -1]), min=0)); q = d_rot / d_nn
    out[f'ratio_{name}'] = dict(median=float(q.median()), q25=float(q.quantile(0.25)), q75=float(q.quantile(0.75)), frac_below_1=float((q < 1).float().mean()))
del G0, Gv, sims
# (2) LDA of the four rotations in the standardised head space
def fit(sub, ridge):
    Xh = np.concatenate([np.asarray(b[sub], np.float32) for b in B], 2)                                 # [n, 4, D]
    mu = Xh.reshape(-1, Xh.shape[2]).mean(0); sd = Xh.reshape(-1, Xh.shape[2]).std(0) + 1e-3
    Z = torch.from_numpy((Xh - mu) / sd).double().to(dev); D = Z.shape[2]; del Xh
    Zf = Z.reshape(-1, D); m = Zf.mean(0); St = (Zf - m).T @ (Zf - m) / Zf.shape[0]
    mc = Z.mean(0); Sb = sum(torch.outer(mc[r] - m, mc[r] - m) for r in range(4)) / 4; del Z, Zf
    Sw = St - Sb + ridge * torch.eye(D, dtype=torch.float64, device=dev)
    ev, U = torch.linalg.eigh(Sw); W = U @ torch.diag(ev.rsqrt()) @ U.T
    e2, V = torch.linalg.eigh(W @ Sb @ W)
    tot = torch.trace(St); pev, pV = torch.linalg.eigh(St)
    res = {'D': int(D), 'between_over_total': float(torch.trace(Sb) / tot), 'lda': {}, 'pca': {}, 'random': {}}
    bases = {}
    for k in (1, 2, 3):
        Lk = torch.linalg.qr(W @ V[:, -k:]).Q; bases[f'lda{k}'] = Lk.float()
        res['lda'][k] = float(torch.trace(Lk.T @ St @ Lk) / tot); res['pca'][k] = float(pev[-k:].sum() / tot)
        rr = []
        for d in range(20):
            Q = torch.linalg.qr(torch.from_numpy(np.random.RandomState(100 + d).randn(D, k)).to(dev)).Q; rr.append(float(torch.trace(Q.T @ St @ Q) / tot))
            if k == 3: bases[f'rand3_{d}'] = Q.float()
        res['random'][k] = dict(mean=float(np.mean(rr)), min=float(np.min(rr)), max=float(np.max(rr)), expected=k / D)
    bases['pca3'] = pV[:, -3:].float()
    return res, bases, mu, sd
VARIANTS = {'n4000_r1e-3': (sub4, 1e-3), 'n4000_r1e-1': (sub4, 1e-1), 'n10000_r1e-3': (sub10, 1e-3), 'n10000_r1e-1': (sub10, 1e-1)}
if not (len(sys.argv) > 5 and sys.argv[5] == 'all'): VARIANTS = {'n4000_r1e-3': VARIANTS['n4000_r1e-3']}
out['subspace'] = {}; KN = {}
def setx(name):
    X = [np.load(f'{ROOT}/runs/ok/s{seed}/{tag}/k0/{name}__L{l}.npy', mmap_mode='r') for l in range(3)]
    return np.concatenate([np.asarray(a[:, 0], np.float32) for a in X], 1)
RAW = {s: setx(s) for s in [IDN, VF] + OODS}
RAWB = np.concatenate([np.asarray(b[bsub, 0], np.float32) for b in B], 1)
def auroc(a, b): r = rankdata(np.concatenate([a, b])); return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))
def knn(q, bk, metric):
    if metric == 'cos':
        qn, bn = torch.nn.functional.normalize(q, dim=1), torch.nn.functional.normalize(bk, dim=1)
        return torch.cat([(qn[i:i + 2000] @ bn.T).topk(10, dim=1).values[:, -1] for i in range(0, len(qn), 2000)]).cpu().numpy()
    return -torch.cat([torch.cdist(q[i:i + 2000], bk).topk(10, dim=1, largest=False).values[:, -1] for i in range(0, len(q), 2000)]).cpu().numpy()
for vn, (sb, ridge) in VARIANTS.items():
    res, bases, mu, sd = fit(sb, ridge); out['subspace'][vn] = res
    Zb = torch.from_numpy((RAWB - mu) / sd).float().to(dev); Zs = {s: torch.from_numpy((x - mu) / sd).float().to(dev) for s, x in RAW.items()}
    proj = {k: v for k, v in bases.items() if k == 'lda3' or vn == 'n4000_r1e-3'}                    # variants: LDA-3 only
    kn = {}
    for pn, P in proj.items():
        bk = Zb @ P
        for metric in (('euc', 'cos') if pn in ('lda3', 'pca3') or pn.startswith('rand3') else ('euc',)):
            si = knn(Zs[IDN] @ P, bk, metric)
            kn[f'{pn}_{metric}'] = {s.replace(f'{split}-', ''): 100 * auroc(si, knn(Zs[s] @ P, bk, metric)) for s in [VF] + OODS}
    for metric in ('euc', 'cos'):                                                                   # random directions: mean and range of 20 draws
        keys = [k for k in kn if k.startswith('rand3_') and k.endswith(f'_{metric}')]
        if keys:
            kn[f'rand3_{metric}'] = {s: dict(mean=float(np.mean([kn[k][s] for k in keys])), min=float(np.min([kn[k][s] for k in keys])),
                                             max=float(np.max([kn[k][s] for k in keys]))) for s in kn[keys[0]]}
            for k in keys: del kn[k]
    KN[vn] = kn; vf = VF.replace(f'{split}-', '')
    print(vn, f'{time.time() - t0:.0f}s', 'flip AUROC', {k: round(kn[k][vf], 1) for k in kn if not k.startswith('rand3')}, flush=True)
    del Zb, Zs, bases
out['knn_projection'] = KN; out['seconds'] = time.time() - t0
os.makedirs(f'{ROOT}/runs/kx/two_subspace2', exist_ok=True)
json.dump(out, open(f'{ROOT}/runs/kx/two_subspace2/{tag}_s{seed}_{split}.json', 'w'), indent=1)
print('done', tag, seed, split, f'{time.time() - t0:.0f}s', flush=True)
