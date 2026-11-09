# checkpoints

## Pretrained models (download into `ckpt/`; used without modification, not redistributed)

`dgcore.py` / `eu/experiments/core.py` (`load_model`) read them from `ckpt/<file>`; the model spec of the job lists is
`<kind>:<file>`. MD5 of the files we used is given for checking.

| tier | `ckpt/` file | spec | model | source | MD5 |
|---|---|---|---|---|---|
| CIFAR-10 | `edm-cifar10-32x32-uncond-vp.pkl` | `edm:` | EDM, VP, DDPM++ (Karras et al., 2022) | https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl | `0cb62d16b8617fd703a697cc17ae422e` |
| SVHN | `layer6_svhn_vpsde.pt` | `l6:` | VP-SDE U-Net (diffusers `UNet2DModel`) of Kamkari et al. (2024) | `layer6ai-labs/dgm_ood_detection`, checkpoint `diffusion_svhn_regretless_ixiwzx2l_final`, renamed | `59c0595951e0064b153c1910f4e37138` |
| CelebA | `diffpath_celeba32.pt` | `iddpm:` | iDDPM of DiffPath (Heng et al., 2024) | https://huggingface.co/ajrheng/diffpath/resolve/main/celeba_ema_0.9999_499999.pt, renamed | `332c7eee2bacf0224df76cdabbb72dce` |
| CIFAR-100 (**removed from the paper**) | `edm_cifar100_uncond_vp.pt` | `edmsd:` | distributed as an unconditional CIFAR-100 DDPM++ with EDM VP preconditioning by Chen et al. (2024) (`huanranchen/DiffusionClassifier`); **behaves as a CIFAR-10 model** — it memorises the CIFAR-10 training images, not CIFAR-100's, and agrees per image with the official CIFAR-10 EDM ([`protocols/CIFAR100_CHECKPOINT.md`](../protocols/CIFAR100_CHECKPOINT.md)) | the authors' Google Drive folder of EDM checkpoints, file `edm_cifar100_uncond_vp.pt` (212.8 MB, dated 2023-11-11) | `bca5663a3c6aa3e0f9d651f3dff3e3f6` |
| ImageNet-64 | `edm2-img64-xs-uncond-2147483-0.100.pkl` | `edm2:` | EDM2-XS, unconditional (Karras et al., 2024) | https://nvlabs-fi-cdn.nvidia.com/edm2/raw-snapshots/edm2-img64-xs-uncond/edm2-img64-xs-uncond-2147483-0.100.pkl | `5b55c856d7fa32c80bc650cbe92e6b28` |
| ImageNet 256 (`in256`, `in256c`) | `256x256_diffusion_uncond.pt` | `adm:256` | ADM, unconditional (Dhariwal & Nichol, 2021; `openai/guided-diffusion`) | https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt | `fd9dd2335b8736d521de0aed54bd90ca` |
| ImageNet 512 (`e512`) | `edm2-img512-xs-uncond-2147483-0.045.pkl` | `edm2:` | EDM2-XS 512, unconditional (the autoguidance models of the EDM2 repository) | the model list of `NVlabs/edm2` | `121ca3ecaf1fe0255a3b77d3a49c0636` |
| ImageNet 256 (`dit256`) | Hugging Face `facebook/DiT-XL-2-256` (snapshot `eab87f77`) | `ditwrap.py` | DiT-XL/2 with its VAE, used with the null class | `huggingface_hub` cache under `data/raw/hf` | — |
| latents of `e512` | Hugging Face `stabilityai/sd-vae-ft-mse` (snapshot `31f26fde`) | `enc_e512.py` | the VAE of EDM2's latent models | `huggingface_hub` cache under `data/raw/hf` | — |

## Models we trained (`runs/models/`, spec `own:models/...`; `eu/experiments/train.py`, EDM preconditioning and loss)

| `runs/models/` | trained on | images seen | used for |
|---|---|---|---|
| `fmnist_s/final.pt` | FashionMNIST train (32×32, zero padded), 22M parameters | 10M | the FashionMNIST tier |
| `eurosat_m/final.pt` | EuroSAT train (32×32), 39M parameters | 4M | EuroSAT (`PROTOCOL_E3.md`) |
| `chest_m/final.pt`, `path_m/final.pt` | ChestMNIST (grey) / PathMNIST train at 32×32 (`code/prep_medmnist.py`) | 8M | `PROTOCOL_E4.md` |
| `c10_m/snap_4M.pt`, `c10_xflip/final.pt`, `c10_rot{25,50,100}/final.pt` | CIFAR-10 train; no augmentation, horizontal flips, random multiples of 90° with probability 0.25 / 0.5 / 1 | 4M | `PROTOCOL_E4.md` |

`train.py` holds out 5,000 training images (2,000 for EuroSAT) (`RandomState(1234)` permutation, stored as `calib_idx` in every checkpoint). The
weights will be published with the per-image scores.
