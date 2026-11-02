# PROTOCOL_UKNN -- orientation-completed kNN ("Upright kNN"): frozen before the single final2 evaluation

Written 2026-11-01 (JST), after development on the selection data (devall-*, sel-*) and the development split (fresh-*); the final2
split (final2-*) has not been read by any script of this family (runs/kx/*): its stored features exist (runs/ok, extracted with all
splits on 2026-10-31) but no score, AUROC or statistic of final2 has been computed for any method.

## Method (all choices below are fixed)

Given a diffusion denoiser D(x; sigma), its ID training images (the kNN bank of PROTOCOL_CT: 50k images, ImageNet-64 10k) and the
shared noise image eps of a seed:

1. Features (one forward pass at sigma_1 = 0.01 of x + sigma_1 eps): outputs of the last decoder blocks at full, half and quarter
   resolution (L0, L1, L2), pooled as [channel mean, channel std] (g) and additionally 2x2-grid [mean, std] (q).
2. Content term (kNN): c_R(x) = R-th largest cosine between the L2-normalised g features of x and of the bank in the space h
   (h in {L1, L012 = mean of the three hook cosines}); (h, R) chosen per ID on the selection data with the content term alone
   (P statistic, R in {1, 2, 5, ..., 1000}).  log pbar(x) = -d log r_R(x), r_R^2 = 2 (1 - c_R(x)), d = Levina-Bickel MLE of the
   intrinsic dimension of the bank in that space (k = 20, 3000 bank points, mean of the per-point estimates).
   Choices (seed-0 selection): CIFAR-10 L012 R=2, SVHN L012 R=5, CelebA L012 R=5, CIFAR-100 L012 R=5, ImageNet-64 L012 R=1000,
   FashionMNIST L012 R=1.
3. Orientation term (ID-only training): the bank images rotated by 0/90/180/270 degrees with the noise NOT rotated
   (R_r(b) + sigma_1 eps), their [L0, L1, L2] x (g, q) features standardised; an MLP (D-512-4, ReLU) trained to predict r
   (AdamW, lr 1e-3 one-cycle, 8 epochs, batch 512 images x 4 rotations); 5000 bank images held out; temperature T fitted on them by NLL.
   log P(up | x) = log softmax(MLP(features of x) / T)_0.   Head variant chosen by the held-out rotation NLL (ID-only):
   MLP on g+q, 512 hidden units, no mirroring.
4. Score (higher = in-distribution): s(x) = -d log r_R(x) + log P(up | x)   (the exact identity log p = log pbar + log P(up|x) + log 4
   for the four-fold rotation group; no weight is tuned).
5. Per noise seed (0, 1, 2) the bank features, d, the head and T are computed with that seed's noise; AUROC averaged over the seeds.

## Rival and comparison rows (unchanged from the held-out comparison)

- tuned kNN: per-ID choice of ok_prof_sel.json (selection data), 3 seeds;
- PROTOCOL_CT kNN (L1, sigma_1, [mean, std], cosine, 10th neighbour), 3 seeds;
- Euphonium (3 seeds), the eight comparison methods (their seeds), RotCls4/RotCls2, Lambda90 where available.

## Evaluation (once)

ours_vs_all.py with SPLIT=final2: per pair the AUROC averaged over the seeds, the bootstrap CI of the difference (2000 replicates,
rng 20261031, ID images resampled once per ID), win / tie / loss (CI excludes 0 and |gap| >= 0.05), 6-ID natural means.
The result is reported as it comes out, whatever it is.

## Development record (before freezing; fresh = development split, already seen)

- Selection data (seed 0, content choice above; head = MLP g+q): CIFAR-10 93.95, SVHN 99.58, CelebA 99.95, CIFAR-100 90.77,
  ImageNet-64 75.45, FashionMNIST 99.98 (tuned kNN: 93.62, 99.38, 99.93, 89.78, 75.44, 99.98).
- fresh, 3 seeds, ana_kx/ours_vs_all.py (SPLIT=fresh): 6-ID natural mean 93.31 (tuned kNN 92.51, PROTOCOL_CT kNN 87.80);
  per pair vs tuned kNN W/T/L 30/31/2 (losses: CIFAR-100 vs CIFAR-10 -2.1, ImageNet-64 vs SSB-hard -0.7); vs the eight comparison
  methods wins 40-62, losses 0-3; upside-down ID sets +0.2 to +29.9.
- Variants seen during development and NOT chosen (reported as ablations from fresh): rotated shared noise in the head's training
  views (the noise pattern reveals the rotation), the orbit bank for the content term, Fisher combination instead of the -d log r sum,
  four-view pose, mirrored bank views for the head (worse held-out rotation NLL), linear head, g-only head inputs, multi-level
  Fisher combinations (L2, sigma_7, rotation-head embedding), learned contrastive heads, two-level Gaussian / kNN aggregations.

## Files (md5 recorded at freezing, GPU-server: ~/upknn/code/)

kx_extract_rot.py, kx_rothead2.py, kx_applyhead.py, kx_export.py, run_final2_uknn.sh; cloud: ana_kx/ours_vs_all.py
