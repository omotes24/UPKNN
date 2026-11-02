# PROTOCOL_HIRES_DIT.md -- addendum to PROTOCOL_HIRES.md (written before any result of the tier): dit256
- Model: DiT-XL/2 256x256 (facebook/DiT-XL-2-256, 675M parameters, class-conditional trained with label dropout) used UNCONDITIONALLY with
  the null class 1000; its own VAE (sd-vae-ft-ema), latents = 0.18215 x posterior mean of the controlled 256 px images (hc_*, the same
  images as in256c), pixels in [-1, 1]; VP mapping of the repository's 1000-step linear schedule (ditwrap.py); fp16.
- Features: DiT has no decoder resolutions, so L0, L1, L2 are fixed as the token outputs of blocks 28, 21, 14 (depth 4/4, 3/4, 2/4),
  reshaped to 16 x 16 maps; everything else is the frozen ImageNet-64 configuration (sigma_1 = 0.01, R = 1000 of the 10,000-image bank,
  d by Levina-Bickel, MLP head on gq features of the four rotated bank views (rotation of the pixels before encoding; noise added to the
  latents, never rotated), temperature by held-out NLL), seeds 0-2.  kNN (basic) on L1 (block 21).  CFS is not defined for a transformer
  (its hook policy needs encoder and decoder blocks) and is not run.

## Implementation note (added 2026-11-02, before any result)
The three seed jobs of dit256 stopped after the feature extraction: the rotation head (code/kx_rothead2.py) ran out of memory on an
11-GB GPU when computing the logits of the 5,000 held-out bank images at once (34,560-dim gq features).  The remaining steps (head,
export, kNN baseline) are run by code/hires_post_dit.sh with code/kx_rothead2_lm2.py, which computes the bank mean / std and the
held-out logits in chunks; same images, same training, same values up to float rounding.  No dit256 score existed before this change.
