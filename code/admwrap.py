# admwrap.py -- the ADM ImageNet 256x256 unconditional model (Dhariwal & Nichol 2021, guided-diffusion, 256x256_diffusion_uncond.pt)
#   as a denoiser D(y, sigma) of the Dragon Glider interface:  y = x + sigma * eps  (pixels in [-1, 1]),  D(y, sigma) = y - sigma * eps_hat.
#   VP mapping: x_t = sqrt(abar_t) * y,  sigma_t^2 = (1 - abar_t) / abar_t  (linear beta schedule, 1000 steps); sigma must be one of the
#   model's sigma_t (sigmas_for() maps a target grid to the nearest integer time and returns the actual sigma_t; the rails are run with
#   those values, so eps_hat = (y - D) / sigma is exactly the model's eps output at x_t = sqrt(abar_t) x + sqrt(1 - abar_t) eps).
#   usage (as a library): den, net = load_adm(device); sig = sigmas_for([0.01, 0.0293, ...])
import os, sys, math
import numpy as np
import torch
ROOT = os.path.expanduser('~/upknn')
sys.path.insert(0, f'{ROOT}/ext/guided-diffusion')
from guided_diffusion.script_util import create_model
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule

CKPT = f'{ROOT}/ckpt/256x256_diffusion_uncond.pt'
BETAS = get_named_beta_schedule('linear', 1000)
ABAR = np.cumprod(1.0 - BETAS)
SIGMA_T = np.sqrt((1.0 - ABAR) / ABAR)              # sigma_0 = 0.01 ... sigma_999 ~ 157


def nearest_t(sigma):
    return int(np.argmin(np.abs(np.log(SIGMA_T) - math.log(float(sigma)))))


def sigmas_for(grid):
    """actual model sigma_t nearest (in log sigma) to each target value"""
    return [float(SIGMA_T[nearest_t(s)]) for s in grid]


def load_adm(device, fp16=True):
    net = create_model(image_size=256, num_channels=256, num_res_blocks=2, channel_mult='', learn_sigma=True, class_cond=False,
                       use_checkpoint=False, attention_resolutions='32,16,8', num_heads=4, num_head_channels=64, num_heads_upsample=-1,
                       use_scale_shift_norm=True, dropout=0.0, resblock_updown=True, use_fp16=fp16, use_new_attention_order=False)
    net.load_state_dict(torch.load(CKPT, map_location='cpu'))
    net.to(device).eval()                 # parameters keep requires_grad: guided-diffusion's attention blocks always checkpoint and
    if fp16: net.convert_to_fp16()        # differentiate w.r.t. their parameters in backward (needed by the PF-ODE divergence of nll.py)
    sig_t = torch.tensor(SIGMA_T, dtype=torch.float64)

    warned = set()

    def fn(x, sigma):
        sg = float(sigma.reshape(-1)[0]) if torch.is_tensor(sigma) else float(sigma)   # one noise level per call (Dragon Glider convention)
        t = nearest_t(sg); s = float(sig_t[t])
        if abs(math.log(s) - math.log(sg)) > 0.05 and t not in warned:      # off-grid sigma (baselines): nearest time, model's own sigma_t
            warned.add(t); print(f'[adm256] sigma {sg:.4g} mapped to t={t} (sigma_t {s:.4g})', flush=True)
        a = 1.0 / math.sqrt(1.0 + s * s)                                   # sqrt(abar_t)
        xt = a * x
        tt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        eps = net(xt, tt)[:, :3].float()                                   # learned-sigma model: first three channels = eps
        return x - s * eps                                                 # E[x0 | x_t] under the model's noise level
    from dgcore import Denoiser
    return Denoiser(fn, 'adm256'), net


def selfcheck(device, n=4):
    """eps_hat from D(y, sigma) against the official q_sample / model call: relative error (should be ~fp16 rounding)"""
    from guided_diffusion.script_util import create_gaussian_diffusion
    den, net = load_adm(device)
    diff = create_gaussian_diffusion(steps=1000, learn_sigma=True, noise_schedule='linear')
    g = torch.Generator(device='cpu').manual_seed(0)
    x0 = (torch.rand([n, 3, 256, 256], generator=g) * 2 - 1).to(device)
    out = {}
    for s in sigmas_for([0.01, 0.0293, 0.0501, 0.1468, 0.4299]):
        t = nearest_t(s); eps = torch.randn([n, 3, 256, 256], generator=g).to(device)
        tt = torch.full((n,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            xt = diff.q_sample(x0, tt, noise=eps); off = net(xt, tt)[:, :3].float()
            ours = (x0 + s * eps - den(x0 + s * eps, s)) / s
        out[s] = float((ours - off).norm() / off.norm())
    return out
