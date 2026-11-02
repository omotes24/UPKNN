# PROTOCOL_HIRES.md -- resolution-controlled ImageNet evaluation at 256x256 (ADM, pixel space) and 512x512 (EDM2-XS uncond, latent space)
#   written before any result of these tiers

Why: at 256x256 (PROTOCOL_IN256.md) SSB-hard and ImageNet-C images (stored at 224 px) are upsampled while ImageNet val (median 375 px)
and the other sets are downsampled; every feature-based detector then separates SSB-hard almost perfectly (a resolution shortcut).
Control: EVERY image (bank, ID, OOD, csID) is first brought to 224 px (prep_newood.crop64 with size 224: box halvings while the short side
>= 448, bicubic to 224 on the short side, centre crop), so that all sets pass through the same resolution bottleneck; then
  - in256c: bicubic 224 -> 256 (uint8, hc_<set>.npy), the ADM 256x256 unconditional model (admwrap, fp16);
  - e512:   bicubic 224 -> 512 on the GPU, Stability VAE (stabilityai/sd-vae-ft-mse, EDM2's encoder, pixels in [0, 1] as in EDM2) posterior MEAN, EDM2 normalisation
            (raw_mean [5.81, 3.25, 0.12, -2.15], raw_std [4.17, 4.62, 3.71, 3.28], final std 0.5), the EDM2-XS ImageNet-512
            UNCONDITIONAL model (edm2-img512-xs-uncond-2147483-0.045.pkl, NVlabs) on the 64x64x4 latents.  Rotations and flips are applied
            to the PIXELS before encoding (np.rot90 / flip); the noise sigma_1 * eps is added to the latents (shared eps, never rotated).
Images: the same images and order as the hr_* arrays of PROTOCOL_IN256.md (hr_<set>_paths.txt); evaluation images = runs/feat/F_in256 idx
(1,000 per set; ID 1,000, its vertical flip, SSB-hard, NINCO, iNaturalist, Textures, OpenImage-O) and the 1,000 csID images per set
(ImageNet-V2, -C, -R).  SSB-hard and ImageNet-C are INCLUDED in these tiers.
Upright kNN: the frozen ImageNet-64 configuration, unchanged (L012 = last decoder block at full, 1/2, 1/4 resolution of the network input;
sigma_1 = 0.01 in the model's input units; R = 1000 of a 10,000-image bank, 10 per class; d by Levina-Bickel (degenerate neighbourhoods
skipped, kx_export_x.py); MLP head on gq features of the four rotated bank views, noise not rotated, 5,000 held out, temperature by held-out
NLL); test sets view 0 only.  Noise seeds 0, 1, 2.
Comparison rows: kNN (basic: L1, 10th neighbour); CFS (lambda 7, official hook policy; 32-image probe in chunks of 4 at 256; fp32), seeds
0-2, both tiers.  in256c only (pixel model, the scripts of the stored 256 statistics with the controlled images; fp32): DiffPath, MSMA*,
GEPC on 500 images per set + 1,000 ID calibration images (same --n/--complement/--cap selections as the stored 256 runs), including
SSB-hard and the csID sets.
Metrics as PROTOCOL_IN256.md (near = SSB-hard, NINCO; far = iNaturalist, Textures, OpenImage-O; full spectrum with ID + csID weighted by
the OpenOOD set sizes; bootstrap 2,000, rng 20261031).  Every number is reported.
