# ditwrap.py -- DiT-XL/2 (Peebles & Xie, ICCV 2023; facebook/DiT-XL-2-256, class-conditional trained with a null class) as an UNCONDITIONAL
#   latent denoiser of the Dragon Glider interface (PROTOCOL_HIRES.md, tier dit256): latents z = 0.18215 * VAE posterior mean (the
#   sd-vae-ft-ema of the DiT repository, pixels in [-1, 1]; enc_dit.py); VP mapping as admwrap.py (the repository's linear betas,
#   1000 steps; sigma_t^2 = (1 - abar_t) / abar_t; nearest t), eps_hat = the first 4 output channels with the null class label 1000
#   (the unconditional branch of classifier-free guidance).
#   Features (an isotropic transformer has no decoder resolutions; fixed before any result): the token outputs of blocks 28, 21, 14
#   (depth 4/4, 3/4, 2/4) reshaped to [B, C, 16, 16], exposed as the probe modules probe0, probe1, probe2 = L0, L1, L2 (net.dg_groups).
import os, math
import numpy as np, torch, torch.nn as nn
ROOT = os.path.expanduser('~/upknn'); HF = f'{ROOT}/data/raw/hf'
from dgcore import Denoiser


class Probe(nn.Module):
    def forward(self, x): return x


def load_dit(device, fp16=True):
    from diffusers import DiTTransformer2DModel, DDPMScheduler
    net = DiTTransformer2DModel.from_pretrained('facebook/DiT-XL-2-256', subfolder='transformer', cache_dir=HF).to(device).eval().requires_grad_(False)
    sch = DDPMScheduler.from_pretrained('facebook/DiT-XL-2-256', subfolder='scheduler', cache_dir=HF)
    abar = np.asarray(sch.alphas_cumprod.double().numpy(), np.float64); sig_t = np.sqrt((1 - abar) / abar)
    G = int(net.config.sample_size) // int(net.config.patch_size)
    for k, b in enumerate((27, 20, 13)):                                   # 0-based indices of blocks 28, 21, 14
        net.add_module(f'probe{k}', Probe())
        def hook(m, i, o, k=k):
            h = o[0] if isinstance(o, (tuple, list)) else o                # [B, N, C] tokens
            B, N, C = h.shape
            getattr(net, f'probe{k}')(h.transpose(1, 2).reshape(B, C, G, G))
        net.transformer_blocks[b].register_forward_hook(hook)
    net.dg_groups = {'L0': ['probe0'], 'L1': ['probe1'], 'L2': ['probe2']}

    def fn(x, sigma):
        sg = float(sigma.reshape(-1)[0]) if torch.is_tensor(sigma) else float(sigma)
        t = int(np.argmin(np.abs(np.log(sig_t) - math.log(sg)))); s = float(sig_t[t]); a = 1.0 / math.sqrt(1.0 + s * s)
        tt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        lab = torch.full((x.shape[0],), 1000, device=x.device, dtype=torch.long)
        with torch.autocast('cuda', dtype=torch.float16, enabled=fp16):
            out = net(a * x, timestep=tt, class_labels=lab).sample
        return x - s * out[:, :x.shape[1]].float()
    return Denoiser(fn, 'dit256'), net
