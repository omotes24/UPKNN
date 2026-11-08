# PROTOCOL_VL.md -- Vertical Limit (VL), fixed before the single final2 evaluation

## Score
For an image x of an ID with the frozen Upright-kNN score s_U(x) = c_t(x) + log q(up | x) (PROTOCOL_UKNN.md, kx_export.py; content term
c_t on L0+L1+L2 at sigma_0 = 0.01 with R_t, MLP pose head with temperature T, per noise seed):

    s_VL(x) = s_U(x) + a0 * [ c_{l*, sigma_0, R*}(x) + log q(up | x) ] + w* * [ c_{l*, sigma*, R*}(x) + log q(up | x) ]

c_{l, sigma, R}(x) = -d log r_R(x): cosine kNN (r = sqrt(2 (1 - cos))) of the L2-normalised [channel mean, channel std] of the output of the
last decoder block at 1/2^l of the input resolution, for the input x + sigma * eps(seed, sigma) (the shared noise image of the seed and
the noise level, dgcore.shared_noise(..., seed, k, 0)); bank = the same ID training images as Upright kNN (50,000; ImageNet-64 10,000);
d = Levina-Bickel dimension of the bank in the same space (k = 20, 3000 bank points drawn with RandomState(0)), per seed.
Every content expert is paired with the same upright posterior log q(up | x) of Upright kNN (product of experts).

## Choices (noise seed 0; no evaluation OOD image is used)  -- code/vl_protocol.py -> runs/vl/PROTOCOL_VL.json
1. l*: the decoder level whose kNN term (R = 1, sigma_0) best separates the ID development images from their own vertical mirror
   images (mirror test, AUROC). Only ID images are used.
2. R*: the R of the grid {1, 2, 5, ..., 5000} with the largest selection-data AUROC (devall ID vs sel- sets) of c_{l*, sigma_0, R} alone
   (the rule Upright kNN used for its R).
3. a0 = 1 if adding the expert [c_{l*, sigma_0, R*} + log q] to s_U raises the selection-data AUROC, else 0.
4. sigma*: the noise level of {sigma_0, ..., sigma_4} (0.01, 0.0171, 0.0293, 0.0501, 0.0855) with the largest mirror test of
   c_{l*, sigma, R*}; w* = max(0, 2 M* - 1) if sigma* != sigma_0, else 0; w* = 0 also when the bootstrap 95% interval of M*
   (2000 resamples) does not exclude 50 %.
Seeds 1 and 2 use the same choices (d recomputed from their banks).

## Evaluation (once)
final2 sets of the six IDs (the same images as the Upright-kNN final2 evaluation), noise seeds 0-2: code/run_vl_final.sh (vl_fextract.py,
vl_fscore.py) -> runs/vl/export/<tag>_s<seed>_final2.npz; ana_kx/vl_vs_uknn.py with SPLIT=final2: per-pair AUROC (mean over seeds),
6-ID natural mean, 63-pair W/T/L against Upright kNN with the bootstrap of ours_vs_all.py (2000, rng 20261031, win/loss when the 95%
interval excludes 0 and |gap| >= 0.05). CPU time: bench_knn/bench_cpu_vl.py (2 threads, CIFAR-10 network; vl = 2 passes, vl0 = 1 pass).
