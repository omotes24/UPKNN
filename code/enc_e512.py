# enc_e512.py -- latents of the e512 tier (PROTOCOL_HIRES.md): hc224 images (pixels / 255, as EDM2 training/encoders.py) -> bicubic 512 -> Stability VAE (sd-vae-ft-mse) posterior mean
#   -> EDM2 normalisation; data/npy/lz512_<name>.npy float16 [N, 64, 64, 4] (NHWC for dgcore.to_t, see run_hires.py) + _src_idx.npy.
#   Names: the evaluation sets of sets_in256c.json (flip applied to the pixels before encoding) and the bank in its four rotations
#   (bank_r0/r90/r180/r270, np.rot90 on the pixels).  Writes runs/ok/sets_e512.json (q512- evaluation sets, rb512- rotated bank).
#   usage: python code/enc_e512.py --gpu 3          (then touches logs/hires/ENC512_DONE)
import os, json, argparse, time
import numpy as np
import torch, torch.nn.functional as F
ROOT = os.path.expanduser('~/upknn'); NPY = f'{ROOT}/data/npy'; HF = f'{ROOT}/data/raw/hf'
p = argparse.ArgumentParser(); p.add_argument('--gpu', type=int, default=0); p.add_argument('--bs', type=int, default=4); a = p.parse_args()
dev = torch.device(f'cuda:{a.gpu}')
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse', cache_dir=HF).to(dev).eval().requires_grad_(False)
RM = torch.tensor([5.81, 3.25, 0.12, -2.15], device=dev).view(1, 4, 1, 1); RS = torch.tensor([4.17, 4.62, 3.71, 3.28], device=dev).view(1, 4, 1, 1)
nfb = 0


@torch.no_grad()
def enc(xu):
    global nfb
    x = torch.from_numpy(np.ascontiguousarray(xu)).to(dev).permute(0, 3, 1, 2).float() / 255          # EDM2 feeds the VAE pixels in [0, 1]
    x = F.interpolate(x, size=(512, 512), mode='bicubic', align_corners=False).clamp(0, 1)
    with torch.autocast('cuda', dtype=torch.float16):
        h = vae.encode(x).latent_dist.mean.float()
    if not torch.isfinite(h).all(): nfb += 1; h = vae.encode(x).latent_dist.mean.float()      # fp32 fallback
    return ((h - RM) / RS * 0.5).permute(0, 2, 3, 1).half().cpu().numpy()


def run(name, xu, src):
    out = f'{NPY}/lz512_{name}.npy'
    if os.path.exists(out): print('exists', name, flush=True); return
    t0 = time.time(); Z = np.concatenate([enc(xu[i:i + a.bs]) for i in range(0, len(xu), a.bs)])
    np.save(out + '.part.npy', Z); os.replace(out + '.part.npy', out); np.save(f'{NPY}/lz512_{name}_src_idx.npy', np.asarray(src, np.int64))
    print(f'[saved] {name} {Z.shape} mean {Z.astype(np.float32).mean((0, 1, 2)).round(3)} std {Z.astype(np.float32).std((0, 1, 2)).round(3)} {time.time() - t0:.0f}s', flush=True)


S = json.load(open(f'{ROOT}/runs/ok/sets_in256c.json')); E = []
for name, data, idx, pre in S:
    xu = np.load(f'{NPY}/{data.replace("hc_", "hc224_")}.npy', mmap_mode='r')[np.asarray(idx, np.int64)]
    if pre == 'vflip': xu = xu[:, ::-1]
    nm = name.replace('q256-', ''); run(nm, xu, idx); E.append([f'q512-{nm}', f'lz512_{nm}', list(range(len(idx))), 'none'])
B = np.load(f'{NPY}/hc224_imagenet_bank.npy', mmap_mode='r')
for k, deg in enumerate((0, 90, 180, 270)):
    run(f'bank_r{deg}', np.rot90(B, k, axes=(1, 2)) if k else B, range(len(B))); E.append([f'rb512-r{deg}', f'lz512_bank_r{deg}', list(range(len(B))), 'none'])
json.dump(E, open(f'{ROOT}/runs/ok/sets_e512.json', 'w'))
open(f'{ROOT}/logs/hires/ENC512_DONE', 'w').write(f'ok fp32 fallbacks {nfb}\n'); print('ENC_E512_DONE fallbacks', nfb, flush=True)
