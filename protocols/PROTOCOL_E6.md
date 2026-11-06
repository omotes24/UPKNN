# PROTOCOL_E6.md -- post-hoc controls for the two-notions analysis of the paper (written 2026-11-06, before any of these numbers was
#   read; no change to the method).  Every number is reported, whatever its sign.  Final2 images, noise seeds 0-2, code/two_controls.py.

Questions (raised by a critical reading of the draft) and what each control answers:
A. Is "kNN barely notices the frame" a property of the pooled whole-image statistics, i.e. would a kNN rule that sees the 2x2 grid
   (the head's own input, standardised) detect upside-down images?  -> 10-NN in the full standardised head space (Euclidean, cosine).
B. Is the "quiet directions" finding an artefact of the discriminant analysis (LDA directions have small within-class scatter by
   construction, and depend on the ridge)?  -> the subspace spanned by the three mean differences between the four rotations needs no
   estimator and no ridge: its variance share (relative to three random directions) and the 10-NN rule restricted to it, next to the
   top-3 principal directions.
C. Does the head need the grid statistics?  -> the same head on the whole-image statistics only (g): accuracy, flip and natural AUROC,
   alone and added to the frozen kNN term.
D. Is the frame read from the diffusion features at all, or would any orientation classifier on simple image statistics do?  -> the
   same head on raw-pixel statistics (mean / SD per channel over the image, the 2x2 grid, and the 4x4 grid), trained on the rotated bank
   images with the same recipe; alone and added to the frozen kNN term.
E. How sensitive is the exponent d to the estimator?  -> Levina-Bickel with k = 5, 10, 20, 50 on 3,000 and 10,000 bank points
   (the paper uses k = 20 on 3,000).  Together with a sweep of the relative weight of the two terms on the development images
   (ana_kx/weight_sweep.py, cloud side) this answers whether the identity's weight is near the optimum and how much d's uncertainty matters.

Expectations written in advance: A detects flips better than the whole-image kNN term where the ratio of Table 1 is below one, but
less well than the head; B behaves like frame-3; C loses some flip and natural AUROC; D detects flips on datasets with a strong
sky-ground statistic (CIFAR-10, ImageNet-64) but reaches a clearly lower natural-OOD AUROC than the paper's head, and adds less to the
kNN term.  If D matches the paper's head, the claim that the frame is read from the frozen features is withdrawn.
