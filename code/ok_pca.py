# ok_pca.py -- compact orbit-response representation for cloud-side exploration (PROTOCOL_OK.md 2026-11-01 01:00 family).
#   Per hook l and noise index k: n_r = L2-normalised global pooled feature (2C) of view r (ok_extract.py, shared noise);
#   harmonics m0 = mean_r n_r, m1 = (n0-n2)/2, m2 = (n0-n1+n2-n3)/4, m3 = (n1-n3)/2 (unnormalised).
#   PCA (centred, fit on the training bank only): n0 -> 128 dims, m1, m2, m3 -> 64 dims each; also the norms |m1|, |m2|, |m3|
#   and the explained-variance ratios.  Output: runs/ok/pca/s<seed>/<tag>/k<k>/L<l>/<name>.npz (Z_n0, Z_m1, Z_m2, Z_m3 float16,
#   nrm [N,3] float32, idx) and pca.npz (means, components, evr).  CPU or GPU (--gpu -1 = CPU).
#   usage: python code/ok_pca.py --tag edmc10 --k 0 --hooks 1 --gpu -1 --prefix devall-,sel-
import os, json, time, argparse
import numpy as np
import torch
ROOT = os.path.expanduser('~/upknn')
p = argparse.ArgumentParser()
p.add_argument('--tag', required=True)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--k', type=int, default=0)
p.add_argument('--hooks', default='1')
p.add_argument('--gpu', type=int, default=-1)
p.add_argument('--prefix', default='devall-,sel-')
a = p.parse_args()
dev = torch.device(f'cuda:{a.gpu}' if a.gpu >= 0 else 'cpu')
if a.gpu < 0: torch.set_num_threads(int(os.environ.get('HARM_THREADS', '12')))
src = f'{ROOT}/runs/ok/s{a.seed}/{a.tag}/k{a.k}'
SETS = [r[0] for r in json.load(open(f'{ROOT}/runs/ok/sets_{a.tag}.json')) if r[0].startswith(tuple(a.prefix.split(',')))]
DIMS = {'n0': 128, 'm1': 64, 'm2': 64, 'm3': 64}
nz = torch.nn.functional.normalize
t0 = time.time()


def harm(X):
    C = X.shape[2] // 10
    n = nz(torch.from_numpy(np.ascontiguousarray(X[:, :, :2 * C])).to(dev).float(), dim=2)
    return {'n0': n[:, 0], 'm1': (n[:, 0] - n[:, 2]) / 2, 'm2': (n[:, 0] - n[:, 1] + n[:, 2] - n[:, 3]) / 4, 'm3': (n[:, 1] - n[:, 3]) / 2}


for l in [int(x) for x in a.hooks.split(',')]:
    od = f'{ROOT}/runs/ok/pca/s{a.seed}/{a.tag}/k{a.k}/L{l}'; os.makedirs(od, exist_ok=True)
    bank = np.load(f'{src}/bank__L{l}.npy', mmap_mode='r')
    parts = [harm(bank[c0:c0 + 5000]) for c0 in range(0, len(bank), 5000)]
    Hb = {h: torch.cat([p_[h] for p_ in parts], 0) for h in DIMS}; del parts
    P = {}
    for h, d in DIMS.items():
        mu = Hb[h].mean(0); Xc = Hb[h] - mu
        cov = Xc.T @ Xc / len(Xc)
        ev, V = torch.linalg.eigh(cov.double()); ev, V = ev.flip(0), V.flip(1)
        P[h] = (mu, V[:, :d].float(), (ev[:d] / ev.sum()).float())
    np.savez(f'{od}/pca.npz', **{f'{h}_mu': P[h][0].cpu().numpy() for h in DIMS}, **{f'{h}_V': P[h][1].cpu().numpy() for h in DIMS},
             **{f'{h}_evr': P[h][2].cpu().numpy() for h in DIMS})

    def save(name, H, idx):
        out = {f'Z_{h}': ((H[h] - P[h][0]) @ P[h][1]).cpu().numpy().astype(np.float16) for h in DIMS}
        out['nrm'] = torch.stack([H[h].norm(dim=1) for h in ('m1', 'm2', 'm3')], 1).cpu().numpy().astype(np.float32)
        np.savez(f'{od}/{name}.npz', idx=idx, **out)

    save('bank', Hb, np.load(f'{src}/bank__idx.npy')); del Hb
    for n in SETS:
        X = np.load(f'{src}/{n}__L{l}.npy', mmap_mode='r')
        save(n, harm(X[:]), np.load(f'{src}/{n}__idx.npy'))
    print(a.tag, a.k, f'L{l}', len(SETS), 'sets', f'{time.time() - t0:.0f}s', {h: round(float(P[h][2].sum()), 3) for h in DIMS}, flush=True)
print('OK_PCA_DONE', a.tag, a.k, f'{time.time() - t0:.0f}s', flush=True)
