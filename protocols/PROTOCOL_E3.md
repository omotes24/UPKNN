# PROTOCOL_E3.md -- Upright kNN on an ID dataset WITHOUT a canonical orientation (EuroSAT, aerial images), written before any result

Question: what does the orientation term do when the ID data have no canonical "up"?  Expectation stated in advance: the held-out
rotation accuracy of the head is low, the calibrated temperature flattens its posterior, and Upright kNN stays close to its kNN term
(no large loss).  Every number of this experiment is reported, whatever its sign.

- Model: the U-Net trained on EuroSAT 32x32 training images (runs/models/eurosat_m/final.pt, trained earlier for another project;
  unchanged).  Tag 'eurosat' registered at run time by code/run_e3.py (SPEC['eurosat'] = (own model, 'eurosat_train', 250)).
- Upright kNN exactly as PROTOCOL_UKNN.md: sigma_1 = 0.01, blocks L0+L1+L2, MLP head on rotated bank views (noise not rotated),
  5,000 bank images held out, temperature by held-out NLL; d by Levina-Bickel.  R = 5, fixed a priori (the median of the R chosen for
  the five 32x32 tiers: 2, 5, 5, 5, 1); no selection on any EuroSAT data.  kx_export_x.py with CH['eurosat'] = ('L012', 5).
- Bank: all 22,000 EuroSAT training images (eurosat_train).
- ID: 1,000 EuroSAT test images (fixed random subset, RandomState(0)), their vertical flips and their 90-degree rotations.
- OOD: the ten natural 32x32 sets of the CIFAR tiers (CIFAR-10, CIFAR-100, SVHN, CelebA, Textures, Tiny-ImageNet, LSUN, ImageNet,
  ImageNet32, Places365), the final-split images of the CIFAR-10 tier (500 each; for CIFAR-10 its final ID images).
- Rows: Upright kNN, its kNN term, its orientation term, kNN (basic: L1, 10th neighbour).  Noise seeds 0, 1, 2.
- Metrics: AUROC per set, mean over the ten OOD sets; AUROC of ID vs its vertical flips / 90-degree rotations (here ~50% is the
  correct behaviour, since EuroSAT has no orientation); held-out rotation accuracy and temperature of the head.
