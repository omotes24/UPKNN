# PROTOCOL_CT -- content + orientation with few forward passes (written 2026-10-28, before any of its statistics exist)

Goal (the user's three conditions, all required): (1) a mean AUROC overwhelmingly above Euphonium, (2) fewer computations
(images per second, measured), (3) no oracle or development-split selection. If any condition fails the result is reported
as a failure.

## History (stated plainly)

- Euphonium (10 passes) has been evaluated on the 63 pairs: 6-ID mean 86.9, best-of-eight on 34 pairs.
- The feature kNN "C" of the final-presentation resume (two decoder blocks at sigma_3, antithetic pair, raw cosine to the
  nearest of 10,000 ID training images) has been evaluated on the 63 pairs: 6-ID mean 90.1, 47 pairs. Its blocks and scale
  were chosen on the development split (development copies of the main OOD datasets), so C itself is not eligible.
- This family was defined after seeing those two results. Every choice inside the family is made on the selection data only.

## Statistics

Content K (new GPU run, `ct_extract.py`):
- Hooks: every U-Net block of the model (the patterns of `nm_featscore.py`); pooled feature of a hook = [channel mean,
  channel std] over the spatial positions of its output.
- Input: x + s_k eps_k and x - s_k eps_k, eps_k the rails' shared noise image of scale index k (seed 0), k = 0..9
  (sigma_1..sigma_10).
- Variant v: plus (the +eps feature; 1 pass) or pair (mean of the +eps and -eps features; 2 passes).
- Hook groups g (from the decoder levels l = log2(R / r) of each model, R = input resolution; "last" = last decoder hook in
  forward order whose output has resolution r): L0, L1, L2 (last decoder block at l = 0, 1, 2), L0+L1, L0+L2, L1+L2,
  L0+L1+L2, DEC (all decoder hooks), ENC (all encoder hooks), ALL (every hook).
- Similarity to a bank image b: mean over the hooks of g of the cosine between the L2-normalised pooled features (same k, v);
  K = the kk-th largest similarity over the bank, kk in {1, 10}.
- Bank: ID training images, nb in {10k, 50k}: a fixed random subset (seed 0) of min(50,000, available) training images; the
  10k bank is the first 10,000 of a fixed permutation (seed 1) of it (nested). ImageNet-64 has 10,000 training images on the
  server, so both of its banks are the same 10,000.

Orientation O (cached rails statistics, no new run): atoms (T, k, v) with log e_k(Tx)/e_k(x), T in {R90, R180, R270},
k = 0..9, v = pair (antithetic runs) or plus (+eps-only runs, same noise image as the pair's +eps); O = sum over atoms.

Score: S = K (content only) or S = z(K) + lambda z(O), lambda in {0.5, 1}, z with the mean and standard deviation of the
1000 ID development images (no OOD data).

Forward passes per image: distinct upright passes (k, sign) plus distinct rotated passes (T, k, sign); content at (k, plus)
uses (k, +), at (k, pair) uses (k, +) and (k, -); an atom (T, k, plus) uses (k, +) and (T, k, +), (T, k, pair) uses both
signs of both. Budget: at most 4 passes (Euphonium: 10). Candidates: content alone, content + one or two atoms.

## Selection data and criterion (the established protocol)

- ID side: the 1000 ID development images common to the pair and +eps-only development runs.
- OOD side: the extra OOD sets of every ID (PROTOCOL_20NFE.md; images common to the pair and +eps-only runs). Never the
  main-evaluation sets, never the development copies of the main OOD datasets.
- Per ID, the mean AUROC over its extra OOD sets; the mean over the six IDs is maximised. Within 0.1 points of the best:
  fewer passes, then the larger worst-ID mean, then the smaller bank, then fewer hooks, then kk = 1.
- The family may be extended (new statistics, new combinations) on the basis of selection-data results only; every
  extension is logged below with its date. The main evaluation is computed once, for the final choice.

## Main evaluation (once)

The chosen configuration on the 63 pairs (the common images of `benchmark/common_idx.json`); 6-ID mean over the natural OOD
sets; pairs that tie (0.05) or beat the best of the eight comparison methods.

## Throughput

GPU-server, one RTX 2080 Ti, the precision of the rails runs, batch 250 (32 px) / 125 (64 px), after warm-up, 2000 images:
images per second of Euphonium (its 10 passes and the log-ratios) and of the chosen configuration (its passes, the pooling,
the nearest-neighbour search over the bank kept on the GPU, the z-sum), for each of the six models.

## Success criteria (fixed now)

- S1 AUROC: 6-ID mean >= Euphonium + 3.0 (>= 89.9); ties or beats the best of the eight on >= 45 of 63 pairs (Euphonium 34);
  no ID mean more than 1.0 below Euphonium's.
- S2 compute: throughput >= 2x Euphonium's on all six models.
- S3 selection: all choices from the selection data; this file and the selection log hashed before the main evaluation;
  one main evaluation.

## Checks (reported, not used for the choice)

Selection without the ImageNet-derived extra sets (STL-10, Imagenette); split-half of the extra sets.

## Log

- 2026-10-28: protocol written; `ct_extract.py` written.
- 2026-10-28: extraction finished (60 jobs: six models x ten scales; runs/ct/<tag>/k0..k9, logs/ct/). Selection: content alone,
  all 800 candidates; content + atoms for all 800 content candidates (every atom set within 4 passes, lambda in {0.5, 1})
  (`select_ct.py --top 800`). While the extraction ran, the content-alone selection AUROCs of CIFAR-100 and CelebA were
  looked at (selection data only). Script hashes (sha256): ct_extract.py c4a91bf2..., mk_ct_sets.py 4b0169e1...,
  ct_pack.py 185fbca0..., ct_extract_defs.py f9e7eef3..., ct_bench.py 14b0ac15....
- 2026-10-28: selection done (144,800 candidates). CHOICE (frozen before any main-evaluation data is read): content alone,
  group L1 (last decoder block at half resolution), sigma_1 (k = 0), plus (1 pass), bank 50k, kk = 10; selection 91.22 (6-ID
  mean; Euphonium 85.75), worst ID 69.1; 9 candidates tied within 0.1 (best 91.30, the same with the pair variant, 2 passes).
  No orientation atom raised the selection mean by more than 0.05. Check without the ImageNet-derived extra sets: the same
  configuration with the 10k bank (tied with 50k). sha256: select_ct.json c85d6f7847668f98..., choice.json 1ecaa5ae1571bdd3...,
  select_ct_noimnet.json 8f677428f169b0b2....
- 2026-10-28: MAIN EVALUATION (once): 6-ID mean 87.56 (Euphonium 86.89, C 90.12), 41 of 63 pairs (Euphonium 34); CIFAR-10 88.6 (-2.6), SVHN 98.0, CelebA 99.9, CIFAR-100 82.0 (+9.2), ImageNet-64 57.6 (-2.9), FashionMNIST 99.2. Throughput 9.2-10.0x Euphonium (runs/ct/bench/). S1 FAILS; S2, S3 pass. No further configuration was evaluated on the main data.
