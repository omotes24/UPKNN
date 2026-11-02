# PROTOCOL_IN256.md -- Upright kNN on ImageNet at 256x256 with the ADM unconditional model (written before any 256 result of Upright kNN)

Purpose: test whether the frozen ImageNet configuration of Upright kNN carries over to a different backbone (ADM ImageNet-256
unconditional, Dhariwal & Nichol 2021) and a 16x larger input (256x256 instead of 64x64), on the OpenOOD v1.5 ImageNet-1K sets.
Nothing below is tuned on 256x256 data; every choice is copied from the ImageNet-64 tier (PROTOCOL_UKNN.md, CH['in64']).

Model and features
- ADM 256x256_diffusion_uncond.pt via code/admwrap.py (fp16), tag 'in256' added at run time by code/run_in256.py
  (SPEC['in256'] = ('adm:256', 'hr_imagenet_bank', 8); no frozen file is edited).
- sigma_1 = 0.01 = the model's smallest noise level (t = 0); one shared noise image per seed (seeds 0, 1, 2), as for every tier.
- Hooks L0, L1, L2 = last decoder block at full, half and quarter resolution (ct_extract_defs.groups_for, unchanged).
- kNN term: L2-normalised [channel mean, channel std] per layer, layers L0+L1+L2 concatenated, cosine similarity to the bank,
  R = 1000 (the ImageNet-64 choice; same bank size of 10,000 images, 10 per class), d by Levina-Bickel (k = 20, 3,000 bank points).
- Orientation term: MLP D-512-4 on the gq descriptors of L0+L1+L2 of the four rotated bank views with the noise NOT rotated
  (kx_extract_rot.py), 8 epochs, 5,000 bank images held out; temperature T by held-out NLL (kx_rothead2.py, unchanged).
- Score = -d log r_R + log q_T(up | x) (kx_export_x.py = kx_export.py + CH['in256'] = ('L012', 1000)).
- Test sets: only the upright view 0 is passed through the model (ok_extract_v0.py, OK_VIEWS=1); the score uses view 0 only.
- The bank's view-0 features are taken from kx_extract_rot.py (view 0 = x + sigma_1 eps, identical input to ok_extract.py's view 0).

Data (all prepared before this protocol; same images as the stored 256x256 comparison methods)
- Bank: hr_imagenet_bank (10,000 ImageNet training images, 10 per class).
- ID: hr_imagenet_test, the 1,000 images of the final split used by the stored 256 statistics (runs/feat/F_in256, idx).
- near-OOD: SSB-hard, NINCO; far-OOD: iNaturalist, Textures, OpenImage-O (hr_*, 1,000 images each, same idx).
- Vertically flipped ID images (same 1,000 ID images, 'vflip').
- Covariate-shifted ID (full spectrum): ImageNet-V2, -C, -R at 256x256 = exactly the 1,000-image subsample of the ImageNet-64
  benchmark (runs/ood1k/s1k), prepared by prep_hr_cs.py with prep_hr.py's crop.

Comparison rows (no refitting of anything that was fitted before)
- kNN (basic): L1, 10th nearest neighbour, cosine (ood1k_knn.py definition), same features.
- CFS (Rouzoumka et al. 2026) at lambda = 7, 1 encoder + 1 decoder hook, diag and kNN heads (cfs_extract.py / cfs_score.py,
  unchanged; fp32 as for the other tiers), seed 0 (+ seeds 1, 2 if time allows).
- MSMA*, DiffPath, GEPC from the stored 256 statistics (runs/base/F_in256, runs/base2/*_in256; fits on the ID calibration images,
  unchanged; 500 images per set). NLL / TT / IC are not available at 256 beyond 50 images (PF-ODE cost) and are not reported.

Metrics
- AUROC (ID positive) and FPR95 per OOD set; near = mean over SSB-hard, NINCO; far = mean over iNaturalist, Textures, OpenImage-O;
  mean over the three noise seeds (sd reported). Full spectrum: ID + csID positive (equal weights per image, 1,000 each).
- Paired comparisons on the common images of each pair of rows; bootstrap 2,000 replicates, rng 20261031; win/loss if the 95% CI of
  the AUROC difference excludes 0 and |gap| >= 0.05 (as everywhere else).
- Every number of this tier is reported, whatever its sign.
