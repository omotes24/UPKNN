# two_controls2.py -- POST-HOC CONTROLS, second set (PROTOCOL_E8.md; written before any result was read; no change to the method).
#   Per ID tier and noise seed, from the stored features and head outputs of the frozen run (final2 images):
#   F. What a low upright posterior means on natural OOD images: for the OOD images flagged by the orientation term only, by the kNN
#      term only, by both and by neither (thresholds keeping 95% of ID images), the head's posterior over the four angles -- its
#      entropy, its maximum, the fraction with max probability < 0.5 ('flat': no recognisable frame) and the fraction peaked (>= 0.5)
#      on a wrong angle ('confidently rotated'); the same for ID images and for upside-down ID images.
#   G. The rotation displacement ratio of Table 1 at fixed neighbour ranks R = 1, 10, 100 for every dataset (the paper uses the
#      protocol's R, which differs across datasets), so that the datasets can be compared at the same R.
#   H. A whitened distance in the head's input space: 10-NN with the Mahalanobis distance of the bank features (covariance of the
#      standardised features with a ridge of 10^-2 and 10^-1 times the mean eigenvalue; 10,000 bank images), flip and natural AUROC --
#      does a distance that whitens see the frame, as the linear read-out does?
#   usage: python code/two_controls2.py <tag> <gpu> [seeds=0,1,2]      -> runs/kx/two_controls2_<tag>_s<seed>.json
import os, sys, json, time
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/upknn')
tag = sys.argv[1]; dev = torch.device(f'cuda:{int(sys.argv[2])}'); SEEDS = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else '0,1,2').split(',')]
SPLIT = 'final2'; R = {'edmc10': 2, 'l6svhn': 5, 'dpceleba': 5, 'edmc100': 5, 'in64': 1000, 'fmnist': 1}[tag]
IDN = f"{SPLIT}-{ {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'edmc100': 'cifar100_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}[tag] }"
SETS = [r for r in json.load(open(f'{ROOT}/runs/ok/sets_{tag}.json')) if r[0].startswith(f'{SPLIT}-')]
VF = [r[0] for r in SETS if r[3] == 'vflip'][0]
OODS = [r[0] for r in SETS if r[3] == 'none' and r[0] != IDN and 'synth' not in r[0]]


def auroc(a, b):
    r = rankdata(np.concatenate([a, b])); return float(100 * (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def summarise(score):
    nat = {s: auroc(score[IDN], score[s]) for s in OODS}
    return dict(flip=auroc(score[IDN], score[VF]), nat=nat, nat_mean=float(np.mean(list(nat.values()))))


def posterior(lg, T):
    z = lg / T; z = z - z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)


def shape(P, flagged=None):
    """posterior statistics of a set of images (optionally only the flagged ones): entropy (nats), fraction flat (max < 0.5),
    fraction peaked on a wrong angle (max >= 0.5 and argmax != 0), fraction peaked on upright, mean P(up)"""
    if flagged is not None: P = P[flagged]
    if len(P) == 0: return dict(n=0)
    H = -(P * np.log(P + 1e-12)).sum(1); mx = P.max(1); am = P.argmax(1)
    return dict(n=int(len(P)), entropy=float(H.mean()), flat=float(np.mean(mx < 0.5)), wrong_peak=float(np.mean((mx >= 0.5) & (am != 0))),
                up_peak=float(np.mean((mx >= 0.5) & (am == 0))), mean_pup=float(P[:, 0].mean()), argmax_hist=[float(np.mean(am == k)) for k in range(4)])


for seed in SEEDS:
    t0 = time.time(); out = {'tag': tag, 'seed': seed, 'split': SPLIT, 'R': R, 'id': IDN, 'vflip': VF, 'ood': OODS}
    EXP = np.load(f'{ROOT}/runs/kx/export/{tag}_s{seed}_{SPLIT}.npz')
    spath = f'{ROOT}/runs/ok/s{seed}/{tag}/k0'
    bpath = f'{ROOT}/runs/kx/rotb/s{seed}/{tag}/k0'; B = [np.load(f'{bpath}/bank__L{l}.npy', mmap_mode='r') for l in range(3)]; N = B[0].shape[0]; Cs = [b.shape[2] // 10 for b in B]
    def setx(name):
        X = [np.load(f'{spath}/{name}__L{l}.npy', mmap_mode='r') for l in range(3)]
        return np.concatenate([np.asarray(a[:, 0, :10 * c], np.float32) for a, c in zip(X, Cs)], 1)
    # the frozen head of the final evaluation (heads.pt: MLP weights, standardisation, temperature), applied as kx_applyhead.py
    ck = torch.load(f'{ROOT}/runs/kx/rothead/s{seed}/{tag}_k0_gqR/heads.pt', map_location='cpu', weights_only=True)
    W1, b1, W2, b2 = (ck['mlp'][k].to(dev).float() for k in ('f1.weight', 'f1.bias', 'f2.weight', 'f2.bias')); hmu, hsd = ck['mu'].to(dev).float(), ck['sd'].to(dev).float(); T = float(ck['T']['mlp'])
    @torch.no_grad()
    def logits(X0):
        z = (torch.from_numpy(np.asarray(X0)).to(dev).float() - hmu) / hsd; return (torch.relu(z @ W1.T + b1) @ W2.T + b2).double().cpu().numpy()
    # ---- F. posterior shape by decision region
    content = {s: EXP[f'content__{s}'].astype(np.float64) for s in [IDN, VF] + OODS}; pose = {s: EXP[f'pose__{s}'].astype(np.float64) for s in [IDN, VF] + OODS}
    P = {}
    for s in [IDN, VF] + OODS:
        assert np.array_equal(np.load(f'{spath}/{s}__idx.npy'), EXP[f'idx__{s}'])
        P[s] = posterior(logits(setx(s)), T)
        assert np.allclose(np.log(P[s][:, 0] + 1e-300), pose[s], atol=1e-2), (s, 'posterior does not match the export', np.abs(np.log(P[s][:, 0] + 1e-300) - pose[s]).max())
    tc, tp = np.quantile(content[IDN], 0.05), np.quantile(pose[IDN], 0.05)
    F_ = {'id': shape(P[IDN]), 'vflip': shape(P[VF]), 'ood_all': {}, 'ood_regions': {}}
    allP = np.concatenate([P[s] for s in OODS]); allc = np.concatenate([content[s] for s in OODS]); allp = np.concatenate([pose[s] for s in OODS])
    fc, fp = allc < tc, allp < tp
    F_['ood_all'] = shape(allP)
    for name, m in (('orient_only', fp & ~fc), ('knn_only', fc & ~fp), ('both', fc & fp), ('neither', ~fc & ~fp)): F_['ood_regions'][name] = shape(allP, m)
    F_['per_set'] = {s: dict(all=shape(P[s]), orient_only=shape(P[s], (pose[s] < tp) & ~(content[s] < tc))) for s in OODS}
    out['F_posterior'] = F_
    # ---- G. displacement ratio at fixed R (kNN space; 4,000 bank images as Table 1)
    def gspace(X): return torch.cat([F.normalize(torch.from_numpy(np.asarray(x[:, :2 * c], np.float32)).to(dev), dim=1) for x, c in zip(X, Cs)], 1) / np.sqrt(3)
    sub4 = np.sort(np.random.RandomState(0).choice(N, min(4000, N), replace=False))
    G0 = gspace([b[:, 0] for b in B]); G2 = gspace([b[sub4, 2] for b in B])
    d_rot = torch.sqrt(torch.clamp(2 * (1 - (G0[sub4] * G2).sum(1)), min=0))
    sims = G0[sub4] @ G0.T; sims[torch.arange(len(sub4), device=dev), torch.from_numpy(sub4).to(dev)] = -2
    top = sims.topk(100, dim=1).values
    out['G_ratio'] = {}
    for Rk in (1, 10, 100, R):
        d_nn = torch.sqrt(torch.clamp(2 * (1 - (top[:, Rk - 1] if Rk <= 100 else sims.topk(Rk, dim=1).values[:, -1])), min=0)); q = d_rot / d_nn
        out['G_ratio'][f'R{Rk}'] = dict(median=float(q.median()), frac_below_1=float((q < 1).float().mean()))
    del G0, G2, sims, top
    # ---- H. whitened (Mahalanobis) 10-NN in the standardised head space
    bsub = np.sort(np.random.RandomState(1).choice(N, min(10000, N), replace=False))
    Xh = np.concatenate([np.asarray(b[bsub, 0], np.float32) for b in B], 1); D = Xh.shape[1]; mu = Xh.mean(0); sd = Xh.std(0) + 1e-3
    Zb = torch.from_numpy((Xh - mu) / sd).to(dev).double(); Zq = {s: torch.from_numpy((setx(s) - mu) / sd).to(dev).double() for s in [IDN, VF] + OODS}
    C = (Zb - Zb.mean(0)).T @ (Zb - Zb.mean(0)) / (len(Zb) - 1); ev, U = torch.linalg.eigh(C); mean_ev = float(ev.mean())
    out['H_whitened'] = {}
    for rel in (1e-2, 1e-1):
        W = (U * (1 / torch.sqrt(ev + rel * mean_ev))).float()                    # x -> x W whitens with ridge
        Bw = (Zb.float() @ W); Qw = {s: (Zq[s].float() @ W) for s in Zq}
        score = {s: -torch.cat([torch.cdist(Qw[s][i:i + 1000], Bw).topk(10, dim=1, largest=False).values[:, -1] for i in range(0, len(Qw[s]), 1000)]).cpu().numpy() for s in Qw}
        out['H_whitened'][f'ridge{rel:g}'] = summarise(score)
    out['seconds'] = time.time() - t0
    json.dump(out, open(f'{ROOT}/runs/kx/two_controls2_{tag}_s{seed}.json', 'w'), indent=1)
    r = F_['ood_regions']
    print(f"TWO_CONTROLS2_DONE {tag} seed {seed} {out['seconds']:.0f}s  F orient-only: n {r['orient_only']['n']} flat {r['orient_only'].get('flat', float('nan')):.2f} wrong {r['orient_only'].get('wrong_peak', float('nan')):.2f}"
          f" | vflip flat {F_['vflip']['flat']:.2f} wrong {F_['vflip']['wrong_peak']:.2f} | id flat {F_['id']['flat']:.2f}  G {out['G_ratio']}  H {[(k, round(v['flip'], 1), round(v['nat_mean'], 1)) for k, v in out['H_whitened'].items()]}", flush=True)
