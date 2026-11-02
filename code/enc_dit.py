# enc_dit.py -- latents of the dit256 tier (PROTOCOL_HIRES.md): hc_ images (256 px, the controlled 224 -> 256 images of in256c; pixels in
#   [-1, 1]) -> the DiT repository's VAE (sd-vae-ft-ema) posterior mean x 0.18215 -> data/npy/lzdit_<name>.npy float16 [N, 32, 32, 4]
#   (NHWC, for run_hires.py's to_t) + _src_idx.npy, for the evaluation sets of sets_in256c.json (flip applied to the pixels) and the bank in
#   its four rotations (np.rot90 on the pixels before encoding).  Writes runs/ok/sets_dit256.json (qdit- sets, rbdit- rotated bank).
#   usage: python code/enc_dit.py --gpu 0          (then touches logs/hires/ENCDIT_DONE)
import os, json, argparse, time
import numpy as np
import torch
ROOT = os.path.expanduser('~/upknn'); NPY = f'{ROOT}/data/npy'; HF = f'{ROOT}/data/raw/hf'
p = argparse.ArgumentParser(); p.add_argument('--gpu', type=int, default=0); p.add_argument('--bs', type=int, default=16); a = p.parse_args()
dev = torch.device(f'cuda:{a.gpu}')
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained('facebook/DiT-XL-2-256', subfolder='vae', cache_dir=HF).to(dev).eval().requires_grad_(False)
SF = float(vae.config.scaling_factor); nfb = 0


@torch.no_grad()
def enc(xu):
    global nfb
    x = torch.from_numpy(np.ascontiguousarray(xu)).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
    with torch.autocast('cuda', dtype=torch.float16):
        h = vae.encode(x).latent_dist.mean.float()
    if not torch.isfinite(h).all(): nfb += 1; h = vae.encode(x).latent_dist.mean.float()
    return (h * SF).permute(0, 2, 3, 1).half().cpu().numpy()


def run(name, xu, src):
    out = f'{NPY}/lzdit_{name}.npy'
    if os.path.exists(out): print('exists', name, flush=True); return
    t0 = time.time(); Z = np.concatenate([enc(xu[i:i + a.bs]) for i in range(0, len(xu), a.bs)])
    np.save(out + '.part.npy', Z); os.replace(out + '.part.npy', out); np.save(f'{NPY}/lzdit_{name}_src_idx.npy', np.asarray(src, np.int64))
    print(f'[saved] {name} {Z.shape} mean {Z.astype(np.float32).mean((0, 1, 2)).round(3)} std {Z.astype(np.float32).std((0, 1, 2)).round(3)} {time.time() - t0:.0f}s', flush=True)


S = json.load(open(f'{ROOT}/runs/ok/sets_in256c.json')); E = []
for name, data, idx, pre in S:
    xu = np.load(f'{NPY}/{data}.npy', mmap_mode='r')[np.asarray(idx, np.int64)]
    if pre == 'vflip': xu = xu[:, ::-1]
    nm = name.replace('q256-', ''); run(nm, xu, idx); E.append([f'qdit-{nm}', f'lzdit_{nm}', list(range(len(idx))), 'none'])
B = np.load(f'{NPY}/hc_imagenet_bank.npy', mmap_mode='r')
for k, deg in enumerate((0, 90, 180, 270)):
    run(f'bank_r{deg}', np.rot90(B, k, axes=(1, 2)) if k else B, range(len(B))); E.append([f'rbdit-r{deg}', f'lzdit_bank_r{deg}', list(range(len(B))), 'none'])
json.dump(E, open(f'{ROOT}/runs/ok/sets_dit256.json', 'w'))
open(f'{ROOT}/logs/hires/ENCDIT_DONE', 'w').write(f'ok fp32 fallbacks {nfb}\n'); print('ENC_DIT_DONE fallbacks', nfb, flush=True)
