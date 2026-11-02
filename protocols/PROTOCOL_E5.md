# PROTOCOL_E5.md -- how Upright kNN and CFS scale with the size of the ImageNet-64 bank; written before any result (2026-11-02).
#   Every number is reported, whatever its sign.  This is an analysis added after the frozen evaluation; the paper's ImageNet-64 numbers
#   (bank of 10 images per class) are not replaced by it.

Question: the ImageNet-64 bank of the frozen protocol has 10,000 images (10 per class, 'in64_imagenet_train10').  Does Upright kNN gain
from a larger bank, and does the comparison with CFS (whose kNN head uses the same bank) change?

Banks: nested random subsets (np.random.RandomState(0).permutation) of the Hugging Face ImageNet-1k 64x64 training shard
(in64_imagenet_train_hf, 256,234 images; ImageNet train split, disjoint from the benchmark's validation images): 10,000 / 25,000 / 50,000
images -> data/npy/in64_hfbank_{10,25,50}k.npy, tags in64h10 / in64h25 / in64h50 (code/mk_e5.py, code/run_e5.py).  The 10k random subset
is not class-balanced; comparing it with the protocol's balanced 10k bank separates the source / balance effect from the size effect.

Upright kNN: the frozen ImageNet-64 configuration otherwise unchanged (EDM2-XS uncond, sigma_1, blocks L0+L1+L2, gq features; MLP head
trained on the four rotated views of THIS bank, 5,000 held out, temperature by held-out NLL; d by Levina-Bickel on 3,000 bank points).
R: primary R = N / 10 (the frozen R/N = 1000/10,000), i.e. 1000 / 2500 / 5000; secondary R = 1000 for every bank (both reported).
kNN (basic: L1, 10th neighbour) on the same bank.  CFS (lambda 7, official hook policy, fp32; its probe images come from the bank) with
its diagonal and kNN heads on the same bank.  Noise seeds 0, 1, 2.

Evaluation: the full OpenOOD v1.5 ImageNet-1K sets at 64x64 (prefix ood1k-; the test features of the frozen run are reused, since they
do not depend on the bank): near (SSB-hard, NINCO), far (iNaturalist, Textures, OpenImage-O), covariate shift (V2, C, R), AUROC and FPR95
(ID positive), full-spectrum AUROC as in the paper; the protocol's own bank (tag in64) as the reference row.

## Implementation note (added 2026-11-02, before any result)
The head is trained with code/kx_rothead2_lm.py: kx_rothead2.py with the bank mean / std of the standardisation computed in chunks
(float64 sums; identical up to float rounding), because the full-bank float32 copy did not fit next to another job on an 11-GB GPU.
