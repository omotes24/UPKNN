# PROTOCOL_E4.md -- Upright kNN on a realistic oriented domain, a second domain without a canonical orientation, and generators trained
#   with rotations; written before any result of these tiers (2026-11-02).  Every number is reported, whatever its sign.

## Questions and expectations stated in advance
A. Chest X-rays (ChestMNIST, 32x32 grey): orientation errors of radiographs (rotated, upside-down, left-right mirrored) are a real
   acquisition / archiving fault.  Expectation: the orientation term detects 90/180-degree rotations and vertical flips; a left-right
   mirror is NOT in the rotation group and is not targeted (reported only).  Other imaging modalities should be detected by the kNN term.
B. Histology (PathMNIST, 32x32 RGB): no canonical orientation (like EuroSAT, PROTOCOL_E3.md).  Expectation: low held-out rotation
   accuracy, a flat calibrated posterior, Upright kNN close to its kNN term.
C. Generators trained with rotations (CIFAR-10, own U-Nets, identical recipe, 4M training images): no augmentation (c10_m at 4M),
   horizontal flips, and a random multiple of 90 degrees with probability 0.25 / 0.5 / 1.0.  With probability 1.0 the generator's
   density is rotation-symmetric by construction (it estimates p-bar of Proposition 1), so its likelihood cannot flag upside-down images.
   Expectation: the likelihood's ID-vs-flip AUROC falls towards 50% as the rotation probability grows, while the orientation term of
   Upright kNN (trained on the upright bank) keeps detecting flips, and log-likelihood + log P(up|x) recovers them (Proposition 1).

## Models (own EDM-type U-Nets, eu/experiments/train.py, arch 'm', 39.4M parameters, bs 128, lr 2.5e-4, warm-up 0.2 Mimg, dropout 0.13)
- chest:   runs/models/chest_m/final.pt  (g32_chest_train, 5,000 calibration images held out; 8 Mimg; the run was interrupted at
           3.07 Mimg on 2026-10-31 and is resumed from its checkpoint with unchanged settings)
- path:    runs/models/path_m/final.pt   (path_train, same recipe, 8 Mimg, resumed likewise)
- c10n4:   runs/models/c10_m/snap_4M.pt  (cifar10_train, no augmentation; the 4-Mimg snapshot of a longer run, same recipe)
- c10x4:   runs/models/c10_xflip/final.pt (xflip 1, 4 Mimg; resumed from its checkpoint at 3.07 Mimg)
- c10r25 / c10r50 / c10r100: runs/models/c10_rot{25,50,100}/final.pt (rotaug 0.25 / 0.5 / 1.0, 4 Mimg)

## Upright kNN: exactly PROTOCOL_UKNN.md / PROTOCOL_E3.md
sigma_1 = 0.01, blocks L0+L1+L2, gq features, MLP head on the four rotated bank views (noise not rotated), 5,000 bank images held out,
temperature by held-out NLL, d by Levina-Bickel (degenerate neighbourhoods skipped), R = 5 fixed a priori (as PROTOCOL_E3.md), no
selection on any data of these tiers.  Bank: ct_extract_defs.bank_indices of the training array (50,000 images; all 50,000 for CIFAR-10).
Noise seeds 0, 1, 2.

## Sets (code/mk_sets_e4.py, prefix e4-; fixed random subsets, RandomState(0))
- chest: ID 1,000 g32_chest_test images, and the same images rotated by 90 / 180 degrees, flipped vertically, mirrored left-right.
         OOD (500 each): other modalities: g32_organa_test (abdominal CT), g32_path_gray_test (histology, grey);
         same modality, other source: g32_pneumonia_all (paediatric chest X-rays; reported separately, not in the means);
         natural: g32_cifar10_gray, g32_svhn_gray, g32_mnist_test, g32_fmnist_test.
- path:  ID 1,000 path_test images and the same four transforms.  OOD (500 each): other modalities: blood_all (blood-cell microscopy),
         derma_all (dermatoscopy), chest_rgb_test (chest X-ray); natural: the ten natural sets of PROTOCOL_E3.md (final2 images of
         the CIFAR-10 tier).
- c10*:  the CIFAR-10 tier's final2 images (ID 500 and their vertical flips; nine natural OOD sets, 500 each), as in the main table.

## Comparison rows
Upright kNN, its kNN term and orientation term, kNN (basic: L1, 10th neighbour), for every tier.
chest, path: CFS (lambda 7, official hook policy, fp32; seeds 0-2), NLL (probability-flow ODE, 64 log-spaced steps), MSMA* and DiffPath
(fitted on 1,000 ID training images outside the bank, the '__cal' run, with the settings of the main benchmark).
c10*: NLL of each generator on the same images, and NLL + log P(up|x) of Upright kNN's head (seed-averaged), as the Proposition-1
completion of the generator's own density.

## Metrics
AUROC (ID positive) per set; means over the groups above; bootstrap of Upright kNN against each row (2,000 replicates, rng 20261031, the
verdict rule of the paper); held-out rotation accuracy and temperature of the head.

## Implementation note (added 2026-11-02, before any result)
The head is trained with code/kx_rothead2_lm.py: kx_rothead2.py with the bank mean / std of the standardisation computed in chunks
(float64 sums; identical up to float rounding), because the full-bank float32 copy did not fit next to another job on an 11-GB GPU.
