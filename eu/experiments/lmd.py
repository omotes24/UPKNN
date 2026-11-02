# lmd.py -- LMD (Lift, Map, Detect; Liu et al., ICML 2023; public code zhenzhel/lift_map_detect) on the denoiser API of core:
#   10 inpaintings per image with alternating 8 x 8 checkerboard masks (recon.py --mask_type checkerboard_alt
#   --mask_num_blocks 8 --reps_per_image 10), each by the predictor-corrector inpainter of controllable_generation.py with the
#   Euler-Maruyama predictor, no corrector and N = 1000 steps of the VP SDE (beta 0.1 -> 20, t from 1 to eps = 1e-3), the known
#   pixels replaced after every step by the forward-noised data at the current time, and the mean of the last step returned;
#   the distance is LPIPS (AlexNet, lpips package) and the score the median over the 10 reconstructions (detect.py).
#   The VP score of the model is (alpha D(x_t / alpha, sigma) - x_t) / (1 - alpha^2), sigma^2 = (1 - alpha^2) / alpha^2.
#   usage: python eu/experiments/lmd.py --model <spec> --tag <tag> --sets <name>:<idx json>,... --gpu 0 [--bs 16]
#   -> runs/lmd/<tag>/<name>.npz with lpips [N, 10] (higher = OOD), idx
import os, sys, json, time, math, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_model, to_t, NPY, ROOT

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--tag', required=True)
p.add_argument('--sets', required=True)
p.add_argument('--steps', type=int, default=1000)
p.add_argument('--reps', type=int, default=10)
p.add_argument('--blocks', type=int, default=8)
p.add_argument('--eps', type=float, default=1e-3)
p.add_argument('--bs', type=int, default=16, help='images per batch (x reps trajectories)')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--gpu', type=int, default=0)
a = p.parse_args()
dev = torch.device(f'cuda:{a.gpu}'); torch.backends.cudnn.benchmark = True
den, net = load_model(a.model, dev)
import lpips
lp = lpips.LPIPS(net='alex', pretrained=True, lpips=True, verbose=False).to(dev).eval()
B0, B1 = 0.1, 20.0
beta = lambda t: B0 + t * (B1 - B0)
logmc = lambda t: -0.25 * t ** 2 * (B1 - B0) - 0.5 * t * B0          # log alpha(t)
outdir = f'{ROOT}/runs/lmd/{a.tag}'; os.makedirs(outdir, exist_ok=True)


def checker(res, ch, inverted):
    bl = res // a.blocks
    m = torch.ones(1, ch, a.blocks, a.blocks)
    if inverted: m[:, :, ::2, ::2] = 0.; m[:, :, 1::2, 1::2] = 0.
    else: m[:, :, 1::2, ::2] = 0.; m[:, :, ::2, 1::2] = 0.
    return m.repeat_interleave(bl, 2).repeat_interleave(bl, 3)


@torch.no_grad()
def inpaint(data, mask, g):
    """data [M,C,H,W] in [-1,1], mask 1 = known; Euler-Maruyama reverse VP SDE with known-pixel replacement"""
    M = data.shape[0]
    x = data * mask + torch.randn(data.shape, generator=g, device=dev) * (1. - mask)
    ts = torch.linspace(1.0, a.eps, a.steps, dtype=torch.float64)
    dt = -1.0 / a.steps
    for i in range(a.steps):
        t = float(ts[i]); al = math.exp(logmc(t)); var = 1.0 - al * al
        sig = math.sqrt(var) / al
        s = (al * den(x / al, sig) - x) / var                            # VP score
        b = beta(t)
        drift = -0.5 * b * x - b * s                                     # reverse-SDE drift f - g^2 s
        x_mean = x + drift * dt
        x = x_mean + math.sqrt(b) * math.sqrt(-dt) * torch.randn(data.shape, generator=g, device=dev)
        md_mean = al * data
        md = md_mean + torch.randn(data.shape, generator=g, device=dev) * math.sqrt(var)
        x = x * (1. - mask) + md * mask
        x_mean = x * (1. - mask) + md_mean * mask
    return x_mean


for item in a.sets.split(','):
    name, jf = item.split(':')
    out = f'{outdir}/{name}.npz'
    if os.path.exists(out): print('exists', out); continue
    idx = np.array(sorted(int(i) for i in json.load(open(jf))), dtype=np.int64)
    data, vflip = (name[:-7], True) if name.endswith('__vflip') else (name.split('__')[0], False)
    xu = np.ascontiguousarray(np.load(f'{NPY}/{data}.npy', mmap_mode='r')[idx])
    dist, t0 = [], time.time()
    for i in range(0, len(xu), a.bs):
        x = to_t(xu[i:i + a.bs], dev)
        if vflip: x = x.flip(2)
        n, C, H, W = x.shape
        X = x.repeat(a.reps, 1, 1, 1)                                   # rows (r, b)
        mask = torch.cat([checker(H, C, r % 2 == 1).expand(n, -1, -1, -1) for r in range(a.reps)]).to(dev)
        g = torch.Generator(device=dev).manual_seed(int(a.seed) * 1000003 + 7919 + i)
        rec = inpaint(X, mask, g).clamp(-1, 1)
        Xl, Rl = (X.repeat(1, 3, 1, 1), rec.repeat(1, 3, 1, 1)) if C == 1 else (X, rec)
        with torch.no_grad(): d = lp(Xl, Rl).flatten().view(a.reps, n).T  # LPIPS takes inputs in [-1, 1]
        dist.append(d.float().cpu().numpy())
        el = time.time() - t0
        print(f'{a.tag} {name} {i + n}/{len(xu)} {el:.0f}s eta {el / (i + n) * (len(xu) - i - n):.0f}s', flush=True)
    dist = np.concatenate(dist)
    np.savez(out, idx=idx, lpips=dist, score=np.median(dist, 1), meta=np.array(str(vars(a))))
    print('saved', out, flush=True)
