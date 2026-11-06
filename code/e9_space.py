# e9_space.py -- PROTOCOL_E9.md, GPU side (written before any result): per ID dataset and noise seed,
#   (i)  d_CFS: the Levina-Bickel intrinsic dimension (20 neighbours of 3,000 random bank descriptors, rng 0, Euclidean) of the
#        standardised CFS descriptor space of runs/cfs_l7 (bank mean / std with the floor of cfs_score.py), and the mean
#        10-NN distance rbar of every final2 set recomputed from the stored descriptors (must equal -knn of the stored scores);
#   (ii) the Gaussian log-density ll(x) = -1/2 sum_j (z_j - mu_j)^2 / v_j of CFS (diag) over all descriptor dimensions, per set;
#   (iii) d_L1: the Levina-Bickel dimension of the space of kNN (basic) (block L1, whole-image [mean, std], unit length, chordal
#        distance; 20 neighbours of 3,000 bank points, rng 0), as kx_export_x.py computes d for the paper's space.
#   usage: python code/e9_space.py <tag> <gpu> [seeds=0,1,2]   -> runs/kx/e9_space_<tag>_s<seed>.npz
import os, sys, json, glob, time
import numpy as np, torch
ROOT = os.path.expanduser('~/upknn'); EPS = 1e-6
tag = sys.argv[1]; dev = torch.device(f'cuda:{int(sys.argv[2])}'); SEEDS = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else '0,1,2').split(',')]


def lb_dim(dist_fn, n_points, k=20, n_sub=3000):
    """Levina-Bickel MLE averaged over n_sub random points (rng 0): dist_fn(sel) -> sorted distances [len(sel), >= k] to the other points"""
    sel = torch.from_numpy(np.random.RandomState(0).choice(n_points, min(n_sub, n_points), replace=False)).to(dev)
    r = dist_fn(sel)[:, :k]
    dd = 1 / torch.log(r[:, -1:] / r[:, :-1]).mean(1); fin = torch.isfinite(dd)
    return float(dd[fin].mean()), int(fin.sum())


for seed in SEEDS:
    t0 = time.time(); out = {}
    # ---- (i), (ii): CFS descriptor space
    cd = f'{ROOT}/runs/cfs_l7/s{seed}/{tag}'
    Bz = torch.from_numpy(np.load(f'{cd}/bank__z.npy')).double().to(dev)
    mu, sd = Bz.mean(0), torch.clamp(Bz.std(0, unbiased=False), min=EPS); v = torch.clamp(Bz.var(0, unbiased=False), min=EPS)
    Bs = ((Bz - mu) / sd).float(); N = len(Bs)
    def cfs_dist(sel):
        d = torch.cdist(Bs[sel], Bs); d[torch.arange(len(sel), device=dev), sel] = float('inf')
        return d.topk(21, dim=1, largest=False).values.double()
    out['d_cfs'], out['d_cfs_n'] = lb_dim(cfs_dist, N)
    with torch.no_grad():
        for f in sorted(glob.glob(f'{cd}/final2-*__z.npy')):
            name = os.path.basename(f)[:-len('__z.npy')]
            Z = torch.from_numpy(np.load(f)).double().to(dev); Zs = ((Z - mu) / sd).float()
            rbar = torch.cat([torch.cdist(Zs[i:i + 1000], Bs).topk(10, dim=1, largest=False).values.mean(1) for i in range(0, len(Zs), 1000)]).double()
            ll = -0.5 * ((Z - mu) ** 2 / v).sum(1)
            out[f'rbar__{name}'] = rbar.cpu().numpy(); out[f'll__{name}'] = ll.cpu().numpy(); out[f'idx__{name}'] = np.load(f'{cd}/{name}__idx.npy')
            S = np.load(f'{cd}/{name}__scores.npz'); assert np.allclose(-S['knn'], out[f'rbar__{name}'], atol=1e-3), (name, 'stored kNN score differs')
    out['D_cfs'] = int(Bs.shape[1]); del Bz, Bs
    # ---- (iii): the space of kNN (basic): L1 block, whole-image statistics, unit length, chordal distance
    X = np.load(f'{ROOT}/runs/ok/s{seed}/{tag}/k0/bank__L1.npy', mmap_mode='r'); C = X.shape[2] // 10
    G = torch.nn.functional.normalize(torch.from_numpy(np.ascontiguousarray(X[:, 0, :2 * C])).to(dev).float(), dim=1)
    def l1_dist(sel):
        c = (G[sel] @ G.T); c[torch.arange(len(sel), device=dev), sel] = -2
        c = c.topk(21, dim=1).values.clamp(max=1 - 1e-6); return torch.sqrt(2 * (1 - c)).double()
    out['d_l1'], out['d_l1_n'] = lb_dim(l1_dist, len(G)); out['N_bank'] = int(len(G)); out['seconds'] = time.time() - t0
    np.savez(f'{ROOT}/runs/kx/e9_space_{tag}_s{seed}.npz', **out)
    print(f"E9_SPACE_DONE {tag} seed {seed} {out['seconds']:.0f}s  d_cfs {out['d_cfs']:.2f} ({out['d_cfs_n']} pts, D {out['D_cfs']})  d_l1 {out['d_l1']:.2f} ({out['d_l1_n']} pts)", flush=True)
