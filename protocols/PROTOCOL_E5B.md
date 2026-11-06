# PROTOCOL_E5B.md -- bank-size scaling of Upright kNN and CFS on ImageNet-64, second run (written before any result; 2026-11-06).
#   Every number is reported, whatever its sign.  Analysis added after the frozen evaluation; the paper's ImageNet-64 numbers
#   (bank of 10 images per class, 'in64_imagenet_train10') are not replaced by it.

## Why a second run
The first run (PROTOCOL_E5.md, tags in64h10/25/50) drew its banks from the Hugging Face shard benjamin-paine/imagenet-1k-64x64, whose
images were produced by a different pipeline (centre crop then Lanczos resize to 64x64) from the benchmark's images (prep_in64.py:
'center-crop-dhariwal', repeated 2x BOX downsampling then BICUBIC).  With those banks every bank-based score, including CFS, fell
below chance against the benchmark's own ImageNet validation images (near-OOD AUROC 45% for Upright kNN, 33% for CFS; e5_ana.json),
i.e. the bank was itself off-distribution.  Those numbers answer a preprocessing question, not the bank-size question, and are kept
in e5/ana/e5_ana.json for the record only.

## Banks (code/mk_e5b.py)
The first 25 / 50 files (sorted names) of every ImageNet-1K training class, preprocessed with the benchmark's crop64 of prep_in64.py
-> data/npy/in64_bank_25k.npy (25,000) and in64_bank_50k.npy (50,000); nested with the protocol bank, which took the first 10 files
per class.  Tags in64b25 / in64b50; the protocol's bank (tag in64, 10,000 images) is the reference row.  The ImageNet training split
is disjoint from the benchmark's validation images.

## Everything else as PROTOCOL_E5.md
Upright kNN: frozen ImageNet-64 configuration (EDM2-XS, sigma_1, blocks L0+L1+L2, gq features; MLP head on the four rotated views of
THIS bank with 5,000 held out, temperature by held-out NLL; d by Levina-Bickel on 3,000 bank points); R = N/10 (primary) and R = 1000.
kNN (basic) on the same bank.  CFS (lambda 7, official hook policy, fp32) with its diagonal and kNN heads on the same bank.
Noise seeds 0, 1, 2.  Evaluation: the full OpenOOD v1.5 ImageNet-1K sets at 64x64 (prefix ood1k-; test features of the frozen run
reused), near / far AUROC and FPR95, csID AUROC, full-spectrum AUROC (e5/ana/e5_ana.py with the new tags).
