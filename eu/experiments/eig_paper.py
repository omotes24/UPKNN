# eig_paper.py -- EigenScore in the setting of its paper (Shoushtari et al., ICLR 2026; arXiv 2510.07206 Table 5 / B.2-B.4):
#   I = 20 noise redraws, K = 3 leading eigenvalues, VP times t in {100, 150, ..., 550} of 1000 (beta 0.1 -> 20), 500 ID + 500 OOD
#   validation images (to choose the times T and the aggregation) and a disjoint 500 + 500 test split, z-scores from training
#   images.  The eigenvalue estimates are those of experiments/baselines.py (Jacobian-free subspace iteration, central differences
#   with step 1% of sigma sqrt(D), three iterations, QR); only the times, the redraws and the image sets differ.
#   usage: SR_FP16=0 python eu/experiments/eig_paper.py --model <spec> --tag <model tag> --sets <name>:<idx json>,... --gpu 0
#   -> runs/eigpaper/<tag>/<name>.npz with lam [N, 10 times, 20 redraws, 3]
import os, sys, json, time, math, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_model, to_t, NPY, ROOT
import baselines as BL

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--tag', required=True)
p.add_argument('--sets', required=True)
p.add_argument('--reps', type=int, default=20)
p.add_argument('--iters', type=int, default=3)
p.add_argument('--neig', type=int, default=3)
p.add_argument('--bs', type=int, default=64)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--gpu', type=int, default=0)
a = p.parse_args()
T = np.arange(100, 551, 50) / 1000.0
BL.EIG_SIG = np.array([math.sqrt(math.expm1(0.5 * 19.9 * t * t + 0.1 * t)) for t in T])
dev = torch.device(f'cuda:{a.gpu}'); torch.backends.cudnn.benchmark = True
den, net = load_model(a.model, dev)
outdir = f'{ROOT}/runs/eigpaper/{a.tag}'; os.makedirs(outdir, exist_ok=True)
for item in a.sets.split(','):
    name, jf = item.split(':')
    out = f'{outdir}/{name}.npz'
    if os.path.exists(out): print('exists', out); continue
    idx = np.array(json.load(open(jf)), dtype=np.int64)
    data, vflip = (name.split('__')[0], False)
    xu = np.ascontiguousarray(np.load(f'{NPY}/{data}.npy', mmap_mode='r')[idx])
    lam, t0 = [], time.time()
    for i in range(0, len(xu), a.bs):
        x = to_t(xu[i:i + a.bs], dev)
        gen = torch.Generator(device='cpu').manual_seed(int(a.seed) * 1000003 + 15485863 + i)
        lam.append(BL.eigen(den, x, gen, a.reps, a.iters, a.neig)['lam'].float().numpy())
        el = time.time() - t0
        print(f'{a.tag} {name} {i + len(x)}/{len(xu)} {el:.0f}s eta {el / (i + len(x)) * (len(xu) - i - len(x)):.0f}s', flush=True)
    np.savez(out, idx=idx, lam=np.concatenate(lam), t=T, sig=BL.EIG_SIG, meta=np.array(str(vars(a))))
    print('saved', out, flush=True)
