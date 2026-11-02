# graham.py -- reconstruction-based detection of Graham et al. (CVPR-W 2023, "Denoising diffusion models for out-of-distribution
#   detection"; public code marksgraham/ddpm-ood) on the denoiser API of core:
#   the test image is noised to L starting levels (the paper: T_start = 10, 20, ..., 1000 of a 1000-step DDPM, i.e. equally spaced
#   times; the public README uses every 4th level = 25 reconstructions), each noised image is reconstructed by a deterministic
#   sampler (the paper: PLMS with 100 steps; here the probability-flow ODE with Heun's method, 50 steps = 99 evaluations, from the
#   starting level down to sigma_min), and the reconstruction is compared with the input by MSE and LPIPS (AlexNet).  The paper
#   z-scores every (level, metric) with validation statistics and averages the z-scores (ood_detection.py); the z-scoring is done
#   in the analysis from the ID development run (--sets <id>__dev128:DEV), so this script stores the raw mse [N, L], lpips [N, L].
#   Starting levels: the sigma_t of the VP SDE (beta 0.1 -> 20) at t = i / L, i = 1..L (equally spaced times, as in the paper);
#   with L = 25 they run from sigma = 0.14 to 152, and the cost is 25 x 99 = 2475 evaluations per image.
#   usage: python eu/experiments/graham.py --model <spec> --tag <tag> --sets <name>:<idx json | DEV>,... --gpu 0 [--levels 25] [--steps 50]
#   -> runs/graham/<tag>/<name>.npz with mse [N, L], lpips [N, L] (higher = OOD), sigmas [L], idx
import os, sys, json, time, math, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_model, to_t, select, NPY, ROOT

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--tag', required=True)
p.add_argument('--sets', required=True)
p.add_argument('--levels', type=int, default=25)
p.add_argument('--steps', type=int, default=50, help='Heun steps per reconstruction (2 * steps - 1 evaluations)')
p.add_argument('--smin', type=float, default=0.002)
p.add_argument('--rho', type=float, default=7.0)
p.add_argument('--bs', type=int, default=64)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--gpu', type=int, default=0)
a = p.parse_args()
dev = torch.device(f'cuda:{a.gpu}'); torch.backends.cudnn.benchmark = True
den, net = load_model(a.model, dev)
import lpips
lp = lpips.LPIPS(net='alex', pretrained=True, lpips=True, verbose=False).to(dev).eval()
B0, B1 = 0.1, 20.0
def vp_sigma(t): return math.sqrt(math.exp(0.5 * t * t * (B1 - B0) + t * B0) - 1.0)         # sigma_t = sqrt(1 / alpha_t^2 - 1)
SIG = [vp_sigma(i / a.levels) for i in range(1, a.levels + 1)]
outdir = f'{ROOT}/runs/graham/{a.tag}'; os.makedirs(outdir, exist_ok=True)
print('start levels', ' '.join('%.3g' % s for s in SIG), flush=True)


@torch.no_grad()
def reconstruct(y, s0):
    """deterministic PF-ODE (Heun, EDM time discretisation) from the noised image y at sigma = s0 down to sigma_min and then 0"""
    n = a.steps
    ts = [(s0 ** (1 / a.rho) + i / (n - 1) * (a.smin ** (1 / a.rho) - s0 ** (1 / a.rho))) ** a.rho for i in range(n)] + [0.0]
    x = y
    for i in range(n):
        s, s1 = ts[i], ts[i + 1]
        d = (x - den(x, s)) / s
        x1 = x + (s1 - s) * d
        if s1 > 0:
            d1 = (x1 - den(x1, s1)) / s1
            x1 = x + (s1 - s) * 0.5 * (d + d1)
        x = x1
    return x


for item in a.sets.split(','):
    name, jf = item.split(':')
    out = f'{outdir}/{name}.npz'
    if os.path.exists(out): print('exists', out); continue
    data, vflip = (name[:-7], True) if name.endswith('__vflip') else (name.split('__')[0], False)
    if jf == 'DEV':                                                      # 128 of the ID development images (first 2000 of the permutation)
        _, didx = select(data, 2000)
        idx = np.sort(np.random.RandomState(2027).choice(didx, 128, replace=False))
    else:
        idx = np.array(sorted(int(i) for i in json.load(open(jf))), dtype=np.int64)
    xu = np.ascontiguousarray(np.load(f'{NPY}/{data}.npy', mmap_mode='r')[idx])
    mse, lpd, t0 = [], [], time.time()
    for i in range(0, len(xu), a.bs):
        x = to_t(xu[i:i + a.bs], dev)
        if vflip: x = x.flip(2)
        m_rows, l_rows = [], []
        for li, s0 in enumerate(SIG):
            g = torch.Generator(device=dev).manual_seed(a.seed * 100003 + li * 1009 + i)
            y = x + s0 * torch.randn(x.shape, generator=g, device=dev)
            r = reconstruct(y, s0).clamp(-1, 1)
            m_rows.append((r - x).double().square().flatten(1).mean(1).cpu().numpy())
            rr = r if r.shape[1] == 3 else r.repeat(1, 3, 1, 1); xx = x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1)
            with torch.no_grad(): l_rows.append(lp(rr.float(), xx.float()).flatten().double().cpu().numpy())
        mse.append(np.stack(m_rows, 1)); lpd.append(np.stack(l_rows, 1))
        el = time.time() - t0
        print(f'{name} {i + len(x)}/{len(xu)} {el:.0f}s', flush=True)
    np.savez(out, idx=idx, mse=np.concatenate(mse), lpips=np.concatenate(lpd), sigmas=np.array(SIG),
             meta=np.array(str(vars(a))))
    print('saved', out, f'{time.time() - t0:.0f}s', flush=True)
