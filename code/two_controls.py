# two_controls.py -- POST-HOC CONTROLS for the two-notions analysis (PROTOCOL_E6.md; written before any result was read; no change to
#   the method).  Per ID tier and noise seed, from the stored features of the frozen run (runs/kx/rotb: four rotated bank views;
#   runs/ok: the final2 evaluation sets, view 0) and from the raw images:
#   A. kNN in the FULL head space (gq statistics of the three blocks, standardised on the bank, 10,000 random bank images, 10th
#      neighbour, Euclidean and cosine): AUROC of ID vs upside-down ID and vs every natural OOD set.  Does a distance that sees the
#      2x2 grid detect the frame without a trained read-out?
#   B. The estimator-free 'mean-3' subspace: the three directions spanned by the differences between the mean features of the four
#      rotations (QR of the 3 difference vectors; no discriminant analysis, no ridge): its share of the total variance, relative to
#      three random directions, and the 10-NN rule restricted to it (as the frame-3 / PC-3 rules of the paper).
#   C. A head on the whole-image statistics only (g: the first 2C of each block; the grid statistics dropped), trained exactly as the
#      paper's head (kx_rothead2.py recipe: MLP D-512-4, AdamW 1e-3 one-cycle, 8 epochs, batch 512 x 4 views, 5,000 held out,
#      temperature by held-out NLL): held-out accuracy, orientation term alone and summed with the frozen kNN term.
#   D. A head on RAW-PIXEL statistics (no denoiser): per channel the mean and SD over the image, over the 2x2 grid ('pix1', 30 dims)
#      and additionally over the 4x4 grid ('pix2', 126 dims), of the bank images rotated in pixel space; same recipe; same outputs.
#      If this head matches the paper's head, the frame is not read from the diffusion features.
#   E. The intrinsic dimension d (Levina-Bickel, kNN space) for k in {5, 10, 20, 50} on 3,000 and 10,000 bank points.
#   usage: python code/two_controls.py <tag> <gpu> [seeds=0,1,2]      -> runs/kx/two_controls_<tag>_s<seed>.json
import os, sys, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ct_extract_defs import SPEC, bank_indices
ROOT = os.path.expanduser('~/upknn'); NPY = f'{ROOT}/data/npy'
tag = sys.argv[1]; dev = torch.device(f'cuda:{int(sys.argv[2])}'); SEEDS = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else '0,1,2').split(',')]
SPLIT = 'final2'; R = {'edmc10': 2, 'l6svhn': 5, 'dpceleba': 5, 'edmc100': 5, 'in64': 1000, 'fmnist': 1}[tag]
SETS = [r for r in json.load(open(f'{ROOT}/runs/ok/sets_{tag}.json')) if r[0].startswith(f'{SPLIT}-')]
IDN = f"{SPLIT}-{ {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'edmc100': 'cifar100_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}[tag] }"
VF = [r[0] for r in SETS if r[3] == 'vflip'][0]; assert IDN in [r[0] for r in SETS], IDN
OODS = [r[0] for r in SETS if r[3] == 'none' and r[0] != IDN and 'synth' not in r[0]]
torch.backends.cudnn.benchmark = True


def auroc(a, b):
    r = rankdata(np.concatenate([a, b])); return float(100 * (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def summarise(score):
    """score: {set name: per-image score (higher = ID)} -> flip AUROC, per-set and mean natural AUROC"""
    nat = {s: auroc(score[IDN], score[s]) for s in OODS}
    return dict(flip=auroc(score[IDN], score[VF]), nat=nat, nat_mean=float(np.mean(list(nat.values()))))


class MLP(nn.Module):
    def __init__(s, D, H=512):
        super().__init__(); s.f1 = nn.Linear(D, H); s.f2 = nn.Linear(H, 4)
    def forward(s, x): return s.f2(F.relu(s.f1(x)))


def train_head(Xb, epochs=8, n_hold=5000):
    """Xb [Nb, 4, D] (float16/32 on dev): the kx_rothead2.py recipe (a linear head is trained first, as in the script, so that the
    random stream is the same); returns the MLP, mu, sd, T, held-out accuracy"""
    torch.manual_seed(0)
    Nb, _, D = Xb.shape
    s1 = torch.zeros(D, dtype=torch.float64, device=dev); s2 = torch.zeros_like(s1); n = 0
    for i in range(0, Nb, 1000):
        Xc = Xb[i:i + 1000].double().reshape(-1, D); s1 += Xc.sum(0); s2 += (Xc ** 2).sum(0); n += len(Xc)
    mu = (s1 / n).float(); sd = ((s2 / n - (s1 / n) ** 2) * n / (n - 1)).clamp(min=0).sqrt().float() + 1e-3
    perm = torch.from_numpy(np.random.RandomState(0).permutation(Nb)).to(dev); va, tr = perm[:n_hold], perm[n_hold:]
    z = lambda X: (X.float() - mu) / sd
    heads = {'lin': nn.Linear(D, 4).to(dev), 'mlp': MLP(D).to(dev)}
    yv = torch.arange(4, device=dev).repeat(len(va))
    for hn, h in heads.items():
        opt = torch.optim.AdamW(h.parameters(), lr=1e-3, weight_decay=1e-4); bs = 512
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * (len(tr) // bs))
        for ep in range(epochs):
            pe = tr[torch.randperm(len(tr), device=dev)]
            for i in range(0, len(pe) - bs + 1, bs):
                b = pe[i:i + bs]; x = z(Xb[b]).reshape(-1, D); y = torch.arange(4, device=dev).repeat(len(b))
                loss = F.cross_entropy(h(x), y); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        h.eval()
    with torch.no_grad():
        lg = torch.cat([heads['mlp'](z(Xb[va[i:i + 1000]]).reshape(-1, D)).float() for i in range(0, len(va), 1000)])
        acc = float((lg.argmax(1) == yv).float().mean())
        Ts = torch.exp(torch.linspace(np.log(0.25), np.log(8), 200, device=dev))
        T = float(Ts[torch.stack([F.cross_entropy(lg / t, yv) for t in Ts]).argmin()])
    return heads['mlp'], mu, sd, T, acc


@torch.no_grad()
def orient_term(head, mu, sd, T, X0):
    lg = head((torch.as_tensor(np.asarray(X0)).to(dev).float() - mu) / sd).double().cpu().numpy() / T
    m = lg.max(1, keepdims=True); return lg[:, 0] - (m[:, 0] + np.log(np.exp(lg - m).sum(1)))


def pix_feats(xu, grids=(1, 2)):
    """raw-pixel statistics of uint8 images [n, H, W, C] (pixels / 255): per channel mean and SD over the image and over each cell
    of a g x g grid for g in grids -> [n, C * 2 * sum(g^2)]"""
    x = torch.from_numpy(np.ascontiguousarray(xu)).to(dev).float() / 255
    if x.ndim == 3: x = x[..., None]
    x = x.permute(0, 3, 1, 2); out = []
    for g in grids:
        H, W = x.shape[2] // g, x.shape[3] // g
        for i in range(g):
            for j in range(g):
                c = x[:, :, i * H:(i + 1) * H, j * W:(j + 1) * W].flatten(2); out += [c.mean(2), c.std(2)]
    return torch.cat(out, 1)


def knn10(Q, Bk, metric):
    if metric == 'cos':
        qn, bn = F.normalize(Q, dim=1), F.normalize(Bk, dim=1)
        return torch.cat([(qn[i:i + 2000] @ bn.T).topk(10, dim=1).values[:, -1] for i in range(0, len(qn), 2000)]).cpu().numpy()
    return -torch.cat([torch.cdist(Q[i:i + 2000], Bk).topk(10, dim=1, largest=False).values[:, -1] for i in range(0, len(Q), 2000)]).cpu().numpy()


spec, bankset, _ = SPEC[tag]
XB = np.load(f'{NPY}/{bankset}.npy', mmap_mode='r'); bidx, _ = bank_indices(len(XB))
setimg = {}
for name, data, idx, pre in SETS:
    if name in [IDN, VF] + OODS: setimg[name] = (data, np.asarray(idx, np.int64), pre)

for seed in SEEDS:
    t0 = time.time(); out = {'tag': tag, 'seed': seed, 'split': SPLIT, 'R': R, 'id': IDN, 'vflip': VF, 'ood': OODS}
    bpath = f'{ROOT}/runs/kx/rotb/s{seed}/{tag}/k0'; spath = f'{ROOT}/runs/ok/s{seed}/{tag}/k0'
    assert np.array_equal(np.load(f'{bpath}/bank__idx.npy'), bidx), 'bank indices differ from ct_extract_defs.bank_indices'
    B = [np.load(f'{bpath}/bank__L{l}.npy', mmap_mode='r') for l in range(3)]; N = B[0].shape[0]; Cs = [b.shape[2] // 10 for b in B]
    def setx(name, part):                              # part 'gq': 10C per block, 'g': first 2C per block (view 0)
        X = [np.load(f'{spath}/{name}__L{l}.npy', mmap_mode='r') for l in range(3)]
        return np.concatenate([np.asarray(a[:, 0, :(10 * c if part == 'gq' else 2 * c)], np.float32) for a, c in zip(X, Cs)], 1)
    EXP = np.load(f'{ROOT}/runs/kx/export/{tag}_s{seed}_{SPLIT}.npz')
    content = {s: EXP[f'content__{s}'].astype(np.float64) for s in [IDN, VF] + OODS}
    for s in [IDN, VF] + OODS: assert np.array_equal(EXP[f'idx__{s}'], np.load(f'{spath}/{s}__idx.npy'))
    # ---- A. kNN in the full (standardised) head space
    bsub = np.sort(np.random.RandomState(1).choice(N, min(10000, N), replace=False))
    Xh = np.concatenate([np.asarray(b[bsub], np.float32) for b in B], 2)                    # [n, 4, D]
    D = Xh.shape[2]; mu = Xh.reshape(-1, D).mean(0); sd = Xh.reshape(-1, D).std(0) + 1e-3
    Zb0 = torch.from_numpy((Xh[:, 0] - mu) / sd).to(dev)
    RAW = {s: torch.from_numpy((setx(s, 'gq') - mu) / sd).to(dev) for s in [IDN, VF] + OODS}
    out['A_full_head_space'] = {m: summarise({s: knn10(RAW[s], Zb0, m) for s in RAW}) for m in ('euc', 'cos')}
    # ---- B. the mean-3 subspace (estimator-free)
    sub4 = np.sort(np.random.RandomState(0).choice(N, min(4000, N), replace=False))
    X4 = np.concatenate([np.asarray(b[sub4], np.float32) for b in B], 2); mu4 = X4.reshape(-1, D).mean(0); sd4 = X4.reshape(-1, D).std(0) + 1e-3
    Z4 = torch.from_numpy((X4 - mu4) / sd4).double().to(dev); Zf = Z4.reshape(-1, D); m = Zf.mean(0)
    St = (Zf - m).T @ (Zf - m) / Zf.shape[0]; tot = float(torch.trace(St))
    mc = Z4.mean(0); M3 = torch.linalg.qr(torch.stack([mc[r] - m for r in range(1, 4)], 1)).Q      # [D, 3]
    share = float(torch.trace(M3.T @ St @ M3)) / tot
    rr = [float(torch.trace(Q.T @ St @ Q)) / tot for Q in (torch.linalg.qr(torch.from_numpy(np.random.RandomState(100 + d).randn(D, 3)).to(dev)).Q for d in range(20))]
    pev, pV = torch.linalg.eigh(St); P3 = pV[:, -3:]
    Zb0_4 = torch.from_numpy((Xh[:, 0] - mu4) / sd4).to(dev).double(); RAW4 = {s: torch.from_numpy((setx(s, 'gq') - mu4) / sd4).to(dev).double() for s in RAW}
    def proj(Q, U): return (Q @ U).float()
    out['B_mean3'] = dict(share_of_total=share, share_over_random3=share / float(np.mean(rr)), random3_mean=float(np.mean(rr)), pca3_share=float(pev[-3:].sum()) / tot,
                          knn={m: summarise({s: knn10(proj(RAW4[s], M3), proj(Zb0_4, M3), m) for s in RAW4}) for m in ('euc', 'cos')},
                          knn_pca3={m: summarise({s: knn10(proj(RAW4[s], P3), proj(Zb0_4, P3), m) for s in RAW4}) for m in ('euc', 'cos')})
    del Z4, Zf, St, Zb0_4, RAW4
    # ---- C. head on g only
    Xg = torch.from_numpy(np.concatenate([np.asarray(b[:, :, :2 * c]) for b, c in zip(B, Cs)], 2)).to(dev)     # [N, 4, 6C] float16
    head, mu_g, sd_g, T, acc = train_head(Xg)
    pose_g = {s: orient_term(head, mu_g, sd_g, T, setx(s, 'g')) for s in RAW}
    out['C_head_g_only'] = dict(acc=acc, T=T, alone=summarise(pose_g), sum_with_knn_term=summarise({s: content[s] + pose_g[s] for s in RAW}))
    del Xg
    # ---- D. heads on raw-pixel statistics
    out['D_pixel_head'] = {}
    for nm, grids in (('pix1', (1, 2)), ('pix2', (1, 2, 4))):
        feats = []
        for k in range(4):
            parts = []
            for i in range(0, len(bidx), 2000):
                xu = np.asarray(XB[bidx[i:i + 2000]]); parts.append(pix_feats(np.rot90(xu, k, axes=(1, 2)), grids))
            feats.append(torch.cat(parts))
        Xp = torch.stack(feats, 1)                                                                # [N, 4, Dp]
        head, mu_p, sd_p, T, acc = train_head(Xp); del Xp
        pose_p = {}
        for s, (data, idx, pre) in setimg.items():
            xu = np.asarray(np.load(f'{NPY}/{data}.npy', mmap_mode='r')[idx])
            if pre == 'vflip': xu = xu[:, ::-1]
            pose_p[s] = orient_term(head, mu_p, sd_p, T, pix_feats(xu, grids).cpu().numpy())
        out['D_pixel_head'][nm] = dict(dim=int(mu_p.shape[0]), acc=acc, T=T, alone=summarise(pose_p), sum_with_knn_term=summarise({s: content[s] + pose_p[s] for s in RAW}))
    # ---- E. d for several k and sample sizes (kNN space of the paper: unit-length [mean, std] per block, concatenated / sqrt 3)
    G = torch.cat([F.normalize(torch.from_numpy(np.asarray(b[:, 0, :2 * c], np.float32)).to(dev), dim=1) for b, c in zip(B, Cs)], 1) / np.sqrt(3)
    out['E_d'] = {}
    for n in (3000, 10000):
        sel = torch.from_numpy(np.random.RandomState(0).choice(N, min(n, N), replace=False)).to(dev)
        c = (G[sel] @ G.T).float().topk(51, dim=1).values[:, 1:].clamp(max=1 - 1e-6); r = torch.sqrt(2 * (1 - c))
        for k in (5, 10, 20, 50):
            dd = 1 / torch.log(r[:, k - 1:k] / r[:, :k - 1]).mean(1); fin = torch.isfinite(dd)
            out['E_d'][f'n{n}_k{k}'] = float(dd[fin].mean())
    out['seconds'] = time.time() - t0
    json.dump(out, open(f'{ROOT}/runs/kx/two_controls_{tag}_s{seed}.json', 'w'), indent=1)
    print(f'TWO_CONTROLS_DONE {tag} seed {seed} {out["seconds"]:.0f}s  A euc flip {out["A_full_head_space"]["euc"]["flip"]:.1f} nat {out["A_full_head_space"]["euc"]["nat_mean"]:.1f}'
          f'  B share/rand {out["B_mean3"]["share_over_random3"]:.2f} flip {out["B_mean3"]["knn"]["euc"]["flip"]:.1f}'
          f'  C acc {100 * out["C_head_g_only"]["acc"]:.1f} alone {out["C_head_g_only"]["alone"]["nat_mean"]:.1f}/{out["C_head_g_only"]["alone"]["flip"]:.1f}'
          f'  D pix2 acc {100 * out["D_pixel_head"]["pix2"]["acc"]:.1f} alone {out["D_pixel_head"]["pix2"]["alone"]["nat_mean"]:.1f}/{out["D_pixel_head"]["pix2"]["alone"]["flip"]:.1f}'
          f'  E d {out["E_d"]}', flush=True)
