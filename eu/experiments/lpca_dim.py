# lpca_dim.py -- model-free LID of the ID training data by local PCA (the calibration target of Kamkari et al.'s threshold;
#   scikit-dimension lPCA, version 'FO', alphaFO = 0.001, n_neighbors = 1000 as in dgm_geometry conf/lid_method/lpca.yaml):
#   for 1000 query images, the number of principal components of their 1000 nearest training images (Euclidean, pixels in [0, 1])
#   whose variance exceeds 0.001 x the largest; reported: the mean over the queries.  Pool: 20,000 training images.
#   usage: python eu/experiments/lpca_dim.py <train array> [pool] [queries]  -> runs/kamkari/lpca_<array>.json
import os, sys, json, time
import numpy as np, torch
ROOT = os.environ.get('SR_ROOT', os.path.expanduser('~/upknn'))
name = sys.argv[1]; P = int(sys.argv[2]) if len(sys.argv) > 2 else 20000; Q = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
torch.set_num_threads(8)
x = np.load(f'{ROOT}/data/npy/{name}.npy', mmap_mode='r')
rs = np.random.RandomState(0); pool = np.sort(rs.choice(len(x), min(P, len(x)), replace=False))
X = torch.from_numpy(np.asarray(x[pool]).reshape(len(pool), -1).astype(np.float32) / 255.0)
qi = rs.choice(len(X), Q, replace=False)
sq = (X * X).sum(1)
dims, t0 = [], time.time()
for n, q in enumerate(qi):
    d2 = sq - 2 * X @ X[q] + sq[q]
    nb = torch.topk(d2, 1000, largest=False).indices
    Z = X[nb] - X[nb].mean(0, keepdim=True)
    ev = torch.linalg.svdvals(Z) ** 2
    dims.append(int((ev > 0.001 * ev[0]).sum()))
    if n % 100 == 0: print(name, n, np.mean(dims), f'{time.time() - t0:.0f}s', flush=True)
out = dict(array=name, pool=int(len(pool)), queries=int(Q), alphaFO=0.001, n_neighbors=1000, mean=float(np.mean(dims)),
           median=float(np.median(dims)), sd=float(np.std(dims)))
os.makedirs(f'{ROOT}/runs/kamkari', exist_ok=True)
json.dump(out, open(f'{ROOT}/runs/kamkari/lpca_{name}.json', 'w'), indent=1)
print(out, flush=True)
