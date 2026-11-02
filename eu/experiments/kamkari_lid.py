# kamkari_lid.py -- local intrinsic dimension of Kamkari et al. (ICML 2024, "A Geometric Explanation of the Likelihood OOD
#   Detection Paradox"; public code layer6ai-labs/dgm_ood_detection lid/diffusions/lid_estimators.py and its refactored
#   version layer6ai-labs/dgm_geometry lid/diffusions/normal_bundle.py): the normal-bundle estimator of Stanczuk et al.
#   For every image x: K = 4 D noised copies at the VP time t0 = 0.01 (beta 0.1 -> 20; the public configuration), their scores
#   s(x_t0, t0) stacked into a K x D matrix, and its singular values (from the float64 Gram matrix; equal to torch.linalg.svdvals
#   to 6e-4 relative on a CIFAR-10 image).  LID(tau) = number of singular values below
#   exp(-2 tau); tau is calibrated offline on 128 training images (analysis), as in the public code.
#   VP time t0 in the EDM convention of core.Denoiser: x_t = alpha (x + sigma eps), sigma^2 = (1 - alpha^2) / alpha^2, and the VP
#   score is s(x_t) = (D(y, sigma) - y) / (alpha sigma^2) with y = x_t / alpha.
#   usage: python eu/experiments/kamkari_lid.py --model <spec> --tag <tag> --sets <name>:<idx json>,... --gpu 0 [--chunk 1024]
#   -> runs/kamkari/<tag>/<name>.npz with sv [N, D] (descending), idx
import os, sys, json, time, math, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_model, to_t, NPY, ROOT

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--tag', required=True)
p.add_argument('--sets', required=True)
p.add_argument('--t0', type=float, default=0.01)
p.add_argument('--kmult', type=float, default=4.0)
p.add_argument('--chunk', type=int, default=1024)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--gpu', type=int, default=0)
a = p.parse_args()
dev = torch.device(f'cuda:{a.gpu}'); torch.backends.cudnn.benchmark = True
den, net = load_model(a.model, dev)
ib = 0.1 * a.t0 + 0.5 * (20.0 - 0.1) * a.t0 ** 2                      # int_0^t0 beta
alpha = math.exp(-0.5 * ib); sigma = math.sqrt((1 - alpha ** 2) / alpha ** 2)
outdir = f'{ROOT}/runs/kamkari/{a.tag}'; os.makedirs(outdir, exist_ok=True)
print(f'{a.tag}: t0 {a.t0} alpha {alpha:.6f} EDM sigma {sigma:.5f}', flush=True)
for item in a.sets.split(','):
    name, jf = item.split(':')
    out = f'{outdir}/{name}.npz'
    if os.path.exists(out): print('exists', out); continue
    idx = np.array(sorted(int(i) for i in json.load(open(jf))), dtype=np.int64)
    data, vflip = (name[:-7], True) if name.endswith('__vflip') else (name.split('__')[0], False)
    xu = np.ascontiguousarray(np.load(f'{NPY}/{data}.npy', mmap_mode='r')[idx])
    svs, t0 = [], time.time()
    for n in range(len(xu)):
        x = to_t(xu[n:n + 1], dev)
        if vflip: x = x.flip(2)
        D = x[0].numel(); K = int(round(a.kmult * D))
        g = torch.Generator(device=dev).manual_seed(int(a.seed) * 1000003 + int(idx[n]))
        G = torch.zeros(D, D, device=dev, dtype=torch.float64)        # Gram matrix S^T S of the K x D score matrix
        with torch.no_grad():
            for c in range(0, K, a.chunk):
                m = min(a.chunk, K - c)
                eps = torch.randn((m,) + tuple(x.shape[1:]), generator=g, device=dev)
                y = x + sigma * eps
                Sc = ((den(y, sigma) - y) / (alpha * sigma ** 2)).flatten(1).double()
                G += Sc.T @ Sc
            sv = torch.linalg.eigvalsh(G).clamp_min(0).sqrt().flip(0)     # singular values of S, descending
        svs.append(sv.cpu().numpy())
        if n % 8 == 0 or n + 1 == len(xu):
            el = time.time() - t0
            print(f'{a.tag} {name} {n + 1}/{len(xu)} {el:.0f}s eta {el / (n + 1) * (len(xu) - n - 1):.0f}s', flush=True)
    np.savez(out, idx=idx, sv=np.stack(svs), t0=a.t0, sigma=sigma, alpha=alpha, K=K, meta=np.array(str(vars(a))))
    print('saved', out, flush=True)
