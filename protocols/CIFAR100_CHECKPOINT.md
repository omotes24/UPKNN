# The CIFAR-100 checkpoint is a CIFAR-10 model (2026-11-09)

The frozen protocol (`PROTOCOL_UKNN.md`) had six ID datasets. The sixth was CIFAR-100 with the public checkpoint
`edm_cifar100_uncond_vp.pt` (223,092,276 bytes, MD5 `bca5663a3c6aa3e0f9d651f3dff3e3f6`; `checkpoints/README.md`), distributed as an
unconditional CIFAR-100 EDM (DDPM++ with the EDM VP preconditioning) in the Google Drive folder "EDM" linked from the README of
`huanranchen/DiffusionClassifier` (Chen et al., 2024), next to `edm_cifar_uncond_vp.pt` of the same size (both dated 2023). Its bank
was the CIFAR-100 training set and its ID test images the CIFAR-100 test set.

After the final evaluation (`final2`, 2026-11-09) we found that the checkpoint behaves as a model trained on CIFAR-10:

1. **Memorisation gap.** Denoising error `||D(x + sigma eps; sigma) - x||^2 / dim` on 1,000 random images of each of CIFAR-10 train,
   CIFAR-10 test, CIFAR-100 train and CIFAR-100 test (one shared noise draw, seed 123), at sigma = 0.05, 0.2, 0.5:

   | model | set | 0.05 | 0.2 | 0.5 |
   |---|---|---|---|---|
   | `edm_cifar100_uncond_vp.pt` | CIFAR-10 train | 0.00066 | 0.00423 | 0.01247 |
   | | CIFAR-10 test | 0.00068 | 0.00445 | 0.01344 |
   | | CIFAR-100 train | 0.00069 | 0.00466 | 0.01451 |
   | | CIFAR-100 test | 0.00069 | 0.00467 | 0.01454 |
   | official `edm-cifar10-32x32-uncond-vp.pkl` | CIFAR-10 train | 0.00066 | 0.00423 | 0.01248 |
   | | CIFAR-10 test | 0.00068 | 0.00445 | 0.01344 |
   | | CIFAR-100 train | 0.00069 | 0.00466 | 0.01451 |
   | | CIFAR-100 test | 0.00069 | 0.00467 | 0.01455 |

   The error on CIFAR-10 *training* images is lower than on CIFAR-10 test images by 0.00022 (sigma 0.2) and 0.00097 (sigma 0.5), the
   same margin as for the official CIFAR-10 EDM (0.00022, 0.00096); on CIFAR-100 the train/test gap is 0.00001 and 0.00003. A model
   trained on CIFAR-100 would show the opposite pattern.
2. **Per-image agreement with the official CIFAR-10 EDM.** The per-image errors of the two models correlate at 0.995 (sigma 0.5) to
   0.999 (sigma 0.05) on every one of the four sets.
3. **Samples and weights** (checked separately, 2026-11-09): 18-step samples from the same noise depict CIFAR-10 classes (frogs, planes,
   trucks, horses) and resemble the official model's samples; of the 10 nearest training images of 256 samples, 82% are CIFAR-10 and
   17% CIFAR-100 images (official CIFAR-10 EDM: 79% / 18%; a CelebA model gives 100% CelebA on the same test). The weights differ from
   the official model's (relative difference 1.50), so the file is an independent CIFAR-10 training run saved under a CIFAR-100 name.

## Decision

A tier whose model was trained on CIFAR-10 while its bank and ID test images are CIFAR-100 does not measure what the benchmark
intends (the bank is not the model's training distribution, and the ID test set is OOD for the model). The CIFAR-100 tier is
therefore removed from every table of the paper and of the supplement, and every aggregate (means over ID datasets, wins / ties /
losses, the near-OOD mean, FPR95, the fusion, plug-in and deep-block analyses, the CSI comparison and the development-split
ablations) is recomputed over the five remaining ID datasets from the per-dataset and per-pair entries of the existing result files
(`paper/gen/fiveid.py`). Nothing is re-evaluated or re-selected; no number of the five remaining datasets changes. The result files
in `paper/gen/data/` keep the CIFAR-100 tier as evaluated (the record of the frozen evaluation). The near-OOD pairs are now three
(CIFAR-10 -> CIFAR-100, SSB-hard, NINCO).

Consequences for the text: the "only loss of Upright kNN to its own kNN term" (CIFAR-100 -> CIFAR-10) and the observation that with
CIFAR-100 as ID the orientation term finds CIFAR-10 images more canonical than CIFAR-100 images were effects of this checkpoint and are
no longer reported; the CSI run with CIFAR-100 as ID (`PROTOCOL_CSI.md`) is dropped with the tier. Six-tier means as evaluated, for the
record: Upright kNN 93.4, its kNN term 92.5, the tuned kNN 92.6, the best prior detector (CFS with a kNN head) 85.5; five-dataset means:
94.9, 94.0, 94.0 and 88.4 (the rotation classifier; CFS with a kNN head 88.1).

A genuine CIFAR-100 tier would need a CIFAR-100-trained diffusion model and a rerun of every comparison method on it; this is left
for future work.
