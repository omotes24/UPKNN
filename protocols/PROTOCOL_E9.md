# PROTOCOL_E9.md -- the frame term as a plug-in to other detectors (written 2026-11-06 before any result; post-hoc analysis, the
#   paper's method and its frozen evaluation are unchanged; every number is reported, whatever its sign).

Question.  Upright kNN adds the log-posterior of the upright pose, log q(R0 | x), to the kNN log-density of our own features, with
the unit weight of the identity log p = log pbar + log P(up | x) + log 4 (the kNN log-density being -d log r_R with d the intrinsic
dimension of that space).  On top of our kNN term the gain is 0.9 points of mean AUROC (92.5 -> 93.4), because that kNN term is
already strong.  Is the frame term a second notion that also improves OTHER neighbourhood detectors, with the same parameter-free
weight, or is its gain specific to our features?

Detectors (per-image scores already stored for the final evaluation images, three noise seeds unless stated):
  (A) CFS (kNN head): score -rbar(x), rbar = mean Euclidean distance to the 10 nearest bank descriptors in CFS's standardised
      descriptor space (cfs_score.py; its official kNN head, bank of 50,000 / 10,000 images as in the paper).
      Upright CFS (kNN):  s(x) = -d_CFS log rbar(x) + log q(R0 | x), d_CFS the Levina-Bickel MLE (20 neighbours of 3,000 random
      bank descriptors, rng 0, Euclidean in the standardised space) in CFS's space; rbar in place of the R-th neighbour distance.
  (B) kNN (basic) of the paper (half-resolution block L1, whole-image statistics, unit-normalised, cosine, 10th neighbour, c_10):
      r(x) = sqrt(2 (1 - c_10(x)));  Upright kNN (basic): s(x) = -d_L1 log r(x) + log q(R0 | x), d_L1 the Levina-Bickel MLE in that
      space (chordal distance, 20 neighbours of 3,000 bank points, rng 0).
  (C) CFS (diag): its score is the mean over the two slots of the mean squared z-score; as a Gaussian log-density it is
      ll(x) = -1/2 sum_j (z_j - mu_j)^2 / v_j over all D descriptor dimensions (same floors as cfs_score.py).
      Upright CFS (diag): s(x) = ll(x) + log q(R0 | x).  A control for the scale argument of Section 3.3: a log-density of thousands
      of dimensions varies by thousands of nats, so the frame factor should change nothing.
  (D) z-sum fusions of the frame term with every comparison method whose scores exist on the ID development images: the eight
      statistics of the model (NLL, IC-png, IC-j2k, TT, MSMA*, DiffPath, GEPC, EigenScore), CFS (diag), CFS (kNN), the rotation
      classifier; z-scores from the ID images of the development split (fresh), as fusion_final2.py; nothing fitted on OOD data.
      DDPM-OOD, LMD and LID+NLL have no development-split ID scores and are not fused.
  q(R0 | x): the paper's frozen head and temperature of each noise seed, unchanged (pose__ of the exports).
Evaluation: the final evaluation images (final2; the common images of ours_vs_all.py), the bootstrap of the paper (2,000
replicates, rng 20261031; win / loss when the 95% interval of the AUROC difference excludes 0 and |gap| >= 0.05), the upright
version against its base per pair; mean AUROC over the natural OOD sets per ID dataset and over the six; flip AUROC (probe).
Three noise seeds for the frame term and for CFS / kNN (basic) (seed-matched); for the fusions the seeds of the base.
GPU side (code/e9_space.py, one job per ID dataset and seed): d_CFS, d_L1 and the Gaussian ll of CFS (diag) per set; everything
else from the stored scores on the CPU (analysis/final/e9/ana/e9_ana.py).

Expectations written in advance.
  E1: Upright CFS (kNN) beats CFS (kNN) on average over the natural OOD sets by several points and wins most of the 47 pairs; the
      gain exceeds the 0.9 points of Upright kNN over its own kNN term, because CFS (kNN) is further below the orientation term's
      89.9% and its scores are weakly correlated with it.
  E2: Upright kNN (basic) beats kNN (basic) (87.7%) in the same way.
  E3: Upright CFS (diag) equals CFS (diag) within 0.2 points of mean AUROC (scale).
  E4: the z-sum fusions improve every detector whose natural-OOD mean is below the orientation term's, the more the lower the base.
If E1 or E2 fail, the frame term's gain is specific to our features and the paper says so.
