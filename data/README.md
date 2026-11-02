# data

The GPU jobs read uint8 arrays `data/npy/<set>.npy` (`[N,32,32,3]`; `[N,32,32]` for the grey-scale sets; `[N,64,64,3]` for
ImageNet-64; `[N,256,256,3]` for the 256×256 sets) and, for the latent models, float16 latents. The arrays are not
redistributed; the scripts below rebuild them from the public sources. The images of every set and split are listed in
`runs/ok/sets_<tier>.json` (`[name, array, indices, transform]`) and `eu/runs/jobs/*_idx/`.

Run from the repository root, in this order (downloads go to `data/raw/`):

| step | writes | sources |
|---|---|---|
| `python code/prep_data.py` | CIFAR-10/100, SVHN, CelebA, Textures (DTD), Tiny-ImageNet val, ImageNet32 val, LSUN and ImageNet (resized; the versions distributed with CSI), Places365 val | public URLs in the script; CIFAR-100, DTD and Places365 (256) are read from `~/datasets/` |
| `python code/prep_gray.py`, `python code/prep_derived.py` | grey-scale sets (FashionMNIST, MNIST, KMNIST, notMNIST, Omniglot) zero padded to 32×32 (`g32_*`), held-out calibration images, uniform-noise and constant images | public URLs; derived arrays |
| `python code/prep_in64.py` | ImageNet-64: ImageNet-1K (OpenOOD v1.5 lists), SSB-hard, NINCO, iNaturalist, OpenImage-O, Textures, Places, CelebA (`in64_*`); the bank `in64_imagenet_train10` (10 training images per class) | OpenOOD v1.5 image lists (`data/openood/benchmark_imglist`) and images under `~/datasets/` |
| `python code/prep_newood.py`, `code/prep_newood3.py`, `code/prep_eurosat.py` | the OOD sets of the selection data (STL-10, GTSRB, Flowers-102, EuroSAT, Oxford-IIIT Pet, Caltech-101, Food-101, FGVC-Aircraft, Imagenette, Rendered SST-2; EMNIST letters, Kuzushiji-49, SEMEION, grey CIFAR-10 / SVHN / GTSRB) | torchvision, Zenodo (EuroSAT), public URLs |
| `python code/prep_ood1k.py` | the OpenOOD v1.5 ImageNet-1K benchmark at 64×64 (ID, near, far, covariate-shifted ID), images used earlier for development or selection excluded | as above |
| `python code/prep_hr.py`, `code/prep_hr_cs.py` | the same sets at 256×256 for ADM (`hr_*`) | as above |
| `python code/prep_c224.py`, `code/mk_sets_hires.py` | resolution control: every image through a 224-px bottleneck, then 256×256 (`hc_*`) | the `hr_*` sources |
| `python code/enc_e512.py`, `code/enc_dit.py` | latents of the controlled images for EDM2-XS 512 (`lz512_*`, Stability VAE `sd-vae-ft-mse`, posterior mean, EDM2 normalisation) and DiT-XL/2 (`lzdit_*`, its own VAE) | the `hc_*` arrays |
| `python code/prep_newid.py`, `code/prep_medmnist.py` | EuroSAT and Tiny-ImageNet training arrays; MedMNIST v2 (ChestMNIST, PathMNIST, OrganAMNIST, PneumoniaMNIST, BloodMNIST, DermaMNIST) at 32×32 | Zenodo (`medmnist` package) |
| `python eu/experiments/prep_in64_hf.py`, `code/mk_e5.py` | ImageNet-1k training images at 64×64 (one shard of `benjamin-paine/imagenet-1k-64x64`) and the nested banks of `PROTOCOL_E5.md` | Hugging Face |
| `python eu/experiments/complexity.py <set>` | PNG / JPEG 2000 code lengths (`data/complexity/`), for the IC comparison method | the arrays |

`~/datasets/` holds the sets that the scripts do not download:

```
cifar100/cifar-100-python/            CIFAR-100, python version
extra_ood/dtd/                        Describable Textures Dataset
places365_val256/val_256/*.jpg        Places365 validation images, 256x256
openood_official/images_largescale/   OpenOOD v1.5: imagenet_1k/{train,val}, inaturalist/extracted, openimage_o/extracted, imagenet_v2, imagenet_c, imagenet_r
openood_official/images_classic/      OpenOOD v1.5: texture, ...
ood_benchmarks/ssb_hard/, ood_benchmarks/ninco/NINCO/NINCO_OOD_classes/
```

The data sets keep their own licences.
