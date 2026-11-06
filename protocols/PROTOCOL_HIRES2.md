# PROTOCOL_HIRES2.md -- ImageNet at 256x256 (ADM, pixel space) with an IDENTICAL resampling history for every image; written before
#   any result (2026-11-06).  Every number is reported, whatever its sign.  Post-hoc control; the paper's numbers are not replaced.

Why: the resolution-controlled run (PROTOCOL_HIRES.md, tier in256c) brings every image to 224 px, but the sets still differ in what
happened before: ImageNet validation images (median short side 375 px) are downsampled by our pipeline, SSB-hard and ImageNet-C images
are stored at 224 px and are not resampled at all.  A denoiser at its finest noise level is sensitive to fine spectral statistics, so a
resampling signature could remain and inflate the near-OOD AUROC (SSB-hard 87.0% in in256c; 68.6% at 64x64).  Two simple pixel
statistics (Laplacian energy, JPEG blockiness) separate SSB-hard from ImageNet with at most 56.7% AUROC, which does not exclude
subtler signatures.

Control (tier in256d): every hc224 image (bank, ID, OOD, csID; the arrays of PROTOCOL_HIRES.md) is downsampled to 112 px by one 2x BOX
reduction and upsampled back to 256 px by one bicubic resize (code/prep_c112.py -> hc112_<set>.npy).  All content above the 112-px
Nyquist limit is removed by the same operation for every set, and every image receives the same upsampling, so no set can be told apart
by how it was resampled before; only the coarse content (112 px) remains.

Expectations written in advance: if the 256-px near-OOD gain reflects image content, the SSB-hard / NINCO AUROC of Upright kNN stays
well above the 64x64 values (68.6 / 71.9, Table 3 of the paper) -- the 112-px images hold more content than 64-px ones; if it reflects a
resampling signature, it falls to those values or below.  The inverted kNN (basic) result on OpenImage-O (23.3% in in256c) should
disappear if it was a resampling artefact.

Everything else as PROTOCOL_HIRES.md: the frozen ImageNet-64 configuration of Upright kNN (sigma_1, L012, R = 1000 of the 10,000-image
bank, d by Levina-Bickel, MLP head on gq of the four rotated bank views with the noise not rotated, 5,000 held out, temperature by held-out
NLL), noise seeds 0-2; kNN (basic); CFS (lambda 7, fp32, official hook policy) with its diagonal and kNN heads, seeds 0-2; the same 1,000
evaluation images per set (SSB-hard, NINCO, iNaturalist, Textures, OpenImage-O, V2, C, R, ID and its vertical flip); AUROC per set, near /
far means, bootstrap as the paper.  The ADM model is used in fp16 for the features (as in in256c).
