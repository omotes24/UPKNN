# hfwrap.py -- Hugging Face diffusers models as denoisers of the Dragon Glider interface, with the VP mapping of admwrap.py:
#   y = x + sigma * eps in the model's data units, sigma_t^2 = (1 - abar_t) / abar_t, nearest t in log sigma, and
#   D(y, sigma) = y - sigma_t * eps_hat(sqrt(abar_t) y, t), so that (y - D) / sigma_t is exactly the model's eps output.
#     'hfddpm:<repo>'  pixel-space DDPM, UNet2DModel + DDPMScheduler (google/ddpm-celebahq-256: linear betas 1e-4 .. 0.02, 1000 steps,
#                      sigma_0 = 0.01 = sigma_1 of the grid)
#     'hfldm:<repo>'   latent diffusion, VQModel + UNet2DModel (CompVis/ldm-celebahq-256: VQ-f4 autoencoder, latents [3, 64, 64],
#                      scaled-linear betas 0.0015 .. 0.0195, sigma_0 = 0.039): the denoiser acts on the latents z = vqvae.encode(x).latents
#                      (before the quantisation, as the model was trained; no scale factor), and den.enc(x) encodes a batch of images
#   Weights are cached under data/raw/hf; fp16 autocast on the GPU unless DG_FP16=0 / SR_FP16=0.
import os, math
import numpy as np
import torch
ROOT = os.path.expanduser('~/upknn'); HF = f'{ROOT}/data/raw/hf'
FP16 = os.environ.get('DG_FP16', os.environ.get('SR_FP16', '1')) == '1'
if f'{ROOT}/code' not in __import__('sys').path: __import__('sys').path.insert(0, f'{ROOT}/code')
from dgcore import Denoiser


class Sched:
    """the discrete VP schedule of a diffusers scheduler config: abar_t, sigma_t and the nearest-time mapping"""
    def __init__(self, repo, subfolder=None):
        from diffusers import DDPMScheduler
        s = DDPMScheduler.from_pretrained(repo, subfolder=subfolder, cache_dir=HF)
        self.config = dict(s.config)
        self.abar = np.asarray(s.alphas_cumprod.double().numpy(), np.float64)
        self.sigma_t = np.sqrt((1.0 - self.abar) / self.abar)

    def nearest_t(self, sigma): return int(np.argmin(np.abs(np.log(self.sigma_t) - math.log(float(sigma)))))

    def sigmas_for(self, grid):
        """the model's sigma_t nearest (in log sigma) to each target value"""
        return [float(self.sigma_t[self.nearest_t(s)]) for s in grid]


def _den(unet, sched, out_ch, name, device):
    ab = torch.tensor(sched.abar, dtype=torch.float64, device=device)
    logsig = 0.5 * (torch.log1p(-ab) - torch.log(ab))
    warned = set()

    def fn(x, sigma):
        t = (logsig[None, :] - sigma.double().log()[:, None]).abs().argmin(1)
        s_t, a_t = logsig[t].exp(), ab[t].sqrt()
        off = float((logsig[t] - sigma.double().log()).abs().max())
        if off > 0.05 and int(t[0]) not in warned:                       # off-grid sigma (baselines): nearest time, model's own sigma_t
            warned.add(int(t[0])); print(f'[{name}] sigma {float(sigma[0]):.4g} mapped to t={int(t[0])} (sigma_t {float(s_t[0]):.4g})', flush=True)
        with torch.autocast('cuda', dtype=torch.float16, enabled=FP16 and x.is_cuda):
            out = unet(a_t.float().view(-1, 1, 1, 1) * x, t).sample
        return x - s_t.float().view(-1, 1, 1, 1) * out[:, :out_ch].float()
    den = Denoiser(fn, name); den.sched = sched
    return den


def load_hf_ddpm(repo, device):
    from diffusers import UNet2DModel
    unet = UNet2DModel.from_pretrained(repo, cache_dir=HF).to(device).eval().requires_grad_(False)        # flat repos (google/ddpm-*)
    return _den(unet, Sched(repo), int(unet.config.in_channels), 'hfddpm:' + repo.split('/')[-1], device), unet


def load_hf_ldm(repo, device):
    from diffusers import UNet2DModel, VQModel
    unet = UNet2DModel.from_pretrained(repo, subfolder='unet', cache_dir=HF).to(device).eval().requires_grad_(False)
    vq = VQModel.from_pretrained(repo, subfolder='vqvae', cache_dir=HF).to(device).eval().requires_grad_(False)
    den = _den(unet, Sched(repo, 'scheduler'), int(unet.config.in_channels), 'hfldm:' + repo.split('/')[-1], device)

    @torch.no_grad()
    def enc(x):
        """images [B, 3, H, W] in [-1, 1] -> latents [B, C, H/f, W/f] before the quantisation (fp32)"""
        return vq.encode(x.float()).latents.float()

    @torch.no_grad()
    def dec(z):
        return vq.decode(z.float()).sample.float()
    den.enc, den.dec, den.vq = enc, dec, vq
    return den, unet


def load_hf(spec, device):
    kind, repo = spec.split(':', 1)
    if kind == 'hfddpm': return load_hf_ddpm(repo, device)
    if kind == 'hfldm': return load_hf_ldm(repo, device)
    raise ValueError(spec)


def selfcheck(device, repo='google/ddpm-celebahq-256', n=2):
    """eps_hat from D(y, sigma) against the scheduler's own forward noising and a direct U-Net call: relative error (~fp16 rounding)"""
    from diffusers import DDPMScheduler
    den, unet = load_hf_ddpm(repo, device)
    s = DDPMScheduler.from_pretrained(repo, cache_dir=HF)
    g = torch.Generator(device='cpu').manual_seed(0)
    x0 = (torch.rand([n, 3, unet.config.sample_size, unet.config.sample_size], generator=g) * 2 - 1).to(device)
    eps = torch.randn(x0.shape, generator=g).to(device)
    for t in (0, 100, 500):
        xt = s.add_noise(x0, eps, torch.tensor([t]))
        with torch.no_grad(): ref = unet(xt, torch.tensor([t] * n, device=device)).sample[:, :3].float()
        sig = float(den.sched.sigma_t[t]); y = x0 + sig * eps
        mine = (y - den(y, sig)) / sig
        print(f't={t} sigma_t={sig:.4g}: max rel diff {(mine - ref).abs().max().item() / ref.abs().max().item():.2e}', flush=True)
