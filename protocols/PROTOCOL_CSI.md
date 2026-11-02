# PROTOCOL_CSI.md -- CSI (Tack et al., NeurIPS 2020) as a comparison method on the five 32x32 ID datasets; written before any result
#   (2026-11-02).  Every number is reported, whatever its sign.  Added after the frozen evaluation (like CFS and the rotation
#   classifiers); the frozen protocol of Upright kNN is not changed by it.

Why: CSI also uses rotation prediction ("shifted instances") to detect novelty and is the closest method that is trained for the task.
It trains a ResNet-18 from scratch on the ID training images for 1,000 epochs (contrastive loss over the rotated views plus a rotation
classification head); Upright kNN reads one pass of a frozen diffusion denoiser and trains only a small head.

Code: the official repository alinlab/CSI at commit 60742b6 (ext/CSI), unmodified, run through code/csi_run.py, which only
 - feeds our uint8 arrays instead of torchvision data sets (float in [0, 1], 3 channels; grey-scale images repeated over channels),
   the same tensors as torchvision's ToTensor of the official CIFAR loader;
 - provides torchlars.LARS (re-implementation of torchlars 0.1.2, which does not build with torch 2.5: weight decay folded into the
   gradient, then scaled by the trust ratio 0.001 |w| / (|g| + wd |w| + 1e-8), weight decay hidden from the base SGD step), and stubs of
   diffdist (used only by multi-GPU runs) and tensorboardX (not installed; the scalars go to a text file);
 - passes the first two entries of image_size = (32, 32, 3) as the output size of the adaptive pooling in RandomResizedCropLayer
   (torch < 1.8 ignored the third entry; newer torch rejects it; the pooling is the identity at 32x32);
 - lets an interrupted run resume from its last checkpoint (utils.load_checkpoint reads config['best'], which the unsupervised modes
   never save; it is set to the value of a fresh start) and never deletes a log directory.
Training (README, unlabeled multi-class setting): --mode simclr_CSI --shift_trans_type rotation --model resnet18 --epochs 1000, all
other settings the defaults of common/common.py (LARS, lr 0.1, 10 warm-up epochs to 10x, cosine schedule, weight decay 1e-6,
temperature 0.5, checkpoint every 10 epochs).  One GPU with --batch_size 128, i.e. the official total batch of 512 (4 GPUs x 32 x 4
rotations; the README runs on one GPU by dropping the distributed launcher; BN statistics over the whole batch as with the official
synchronised BN).  One run per ID; the official code sets no training seed.  The model after the last epoch is used.
Training images: the bank of the tier (ct_extract_defs.bank_indices), i.e. the training images of the kNN bank and of the rotation head
of Upright kNN: all 50,000 training images for CIFAR-10 and CIFAR-100, a fixed random 50,000 of the training sets of SVHN, CelebA and
FashionMNIST.  IDs (tags): CIFAR-10 (edmc10), CIFAR-100 (edmc100), SVHN (l6svhn), CelebA (dpceleba), FashionMNIST (fmnist).
ImageNet-64 is not included: the official code has no 64x64 multi-class pipeline (its ImageNet path is ImageNet-30 at 224 px).
Score: the official 'CSI' score of evals/ood_pre.py with the README's evaluation flags (--ood_samples 10 --resize_factor 0.54
--resize_fix): for each of the four rotations, the largest cosine similarity to the training images times the feature norm, plus the
rotation logit, weighted by the normalisers computed on the training images, averaged over 10 augmented samples.  Computed once,
after training, for every image of every final2 set of the tier (runs/ok/sets_<tag>.json; 500 images per set: ID test images, the
natural OOD sets, the upside-down ID images, noise and constant images).  Higher = more ID.  Output: runs/csi/scores/<tag>/<set>.npz.
Analysis (final/csi/ana/csi_ana.py, written before the scores are read): AUROC (ID positive) on the common images of the frozen
evaluation; bootstrap comparison with Upright kNN exactly as in ours_vs_all.py (2,000 replicates, rng 20261031 with the same draw order,
ID images resampled once per ID and OOD images per set, Upright kNN averaged over noise seeds 0-2, CSI one run; win / loss if the 95%
interval of AUROC(Upright kNN) - AUROC(CSI) excludes 0 and the gap is at least 0.05 points); counts over all pairs and over the natural
pairs; mean over the natural OOD sets per ID with its bootstrap interval; the upside-down set separately; FPR at 95% TPR.  Reproduction
check: CIFAR-10 -> CIFAR-100 against the published 89.3% of the unlabeled multi-class model (repository README), reported whatever the
value.  Cost: GPU-hours of training (from the logs) and GPU seconds per image of scoring (40 ResNet-18 passes per image).
No change after any result is seen: no other hyperparameters, epochs, checkpoints or training runs.

## Implementation note (added 2026-11-02, before any result)
Smoke test of code/csi_run.py on 640 CIFAR-10 bank images (CSI_SMOKE=640, CSI_RUN=runs/csi_smoke, batch 32, 20 epochs, interrupted
and resumed to 30 epochs, then scored on the final2 sets): runs, resumes and scores; peak GPU memory 1.5 GiB at batch 32.  These
outputs (runs/csi_smoke/) are a code test only and are not used in any result.
