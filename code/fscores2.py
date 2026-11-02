# fscores2.py -- diagonal-Gaussian typicality of pooled denoiser activations for every orientation of every image
#   usage: python fscores2.py <bank4.npz> <featdir> <outdir>
#   bank4 holds the training-image features for all four orientations; two reference models are fitted per (hook, sigma):
#     up  : mean / variance of the upright training features
#     sym : mean / variance over all four orientations (features of the rotation-symmetrised training distribution)
#   output per set: d{up|sym}_h{h}_s{s} [n, 4] = minus the mean squared standardised deviation, columns = id, rot90, rot180, rot270
import sys, os, glob
import numpy as np

bankf, fd, od = sys.argv[1:4]
os.makedirs(od, exist_ok=True)
B = np.load(bankf)
ref = {}
for h in ('f0', 'f1'):
    for si in range(B[h].shape[1]):
        up = B[h][:, si, 0].astype(np.float64); al = B[h][:, si].reshape(-1, B[h].shape[-1]).astype(np.float64)
        ref[(h, si)] = {'up': (up.mean(0), up.std(0) + 1e-6), 'sym': (al.mean(0), al.std(0) + 1e-6)}
for f in sorted(glob.glob(f'{fd}/*.npz')):
    name = os.path.basename(f)[:-4]
    if name.startswith('bank') or os.path.exists(f'{od}/{name}.npz'): continue
    z = np.load(f); res = {'idx': z['idx']}
    for (h, si), r in ref.items():
        F = z[h][:, si].astype(np.float64)                               # [n, T, C]
        for k, (mu, sd) in r.items():
            res[f'd{k}_h{h[1]}_s{si}'] = (-(((F - mu) / sd) ** 2).mean(-1)).astype(np.float32)
    np.savez(f'{od}/{name}.npz', **res); print('scored', name, flush=True)
print('FSCORES2_DONE', fd)
