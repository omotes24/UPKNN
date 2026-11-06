# PROTOCOL_E8.md -- second set of post-hoc controls for the two-notions analysis (written 2026-11-06, before any of these numbers
#   was read; no change to the method).  Every number is reported, whatever its sign.  Final2 images, noise seeds 0-2, code/two_controls2.py.

F. What does a low upright posterior mean on a natural OOD image?  Two readings are possible: the head finds no recognisable frame
   (a flat posterior over the four angles) or it is confident that the image is rotated (a posterior peaked on a wrong angle).  For
   the OOD images flagged by the orientation term only, by the kNN term only, by both and by neither (thresholds keeping 95% of ID
   images), the entropy of the frozen head's posterior, the fraction with maximum probability below 0.5 ('flat') and the fraction
   peaked on a non-upright angle; the same for ID images and for upside-down ID images (which should be peaked on 180 degrees).
G. The displacement ratio of Table 1 at fixed neighbour ranks R = 1, 10, 100 for every dataset, so that the six datasets can be
   compared at the same R (the paper uses the protocol's R, which ranges from 1 to 1000).
H. A whitened distance in the head's input space: 10-NN with the Mahalanobis distance of the standardised bank features (ridge of
   0.01 and 0.1 times the mean eigenvalue; 10,000 bank images): flip and natural-OOD AUROC.  If whitening lets a distance see the
   frame, the read-out's advantage over the kNN rule is one of metric, not of being a classifier.

Expectations written in advance: F: upside-down ID images peaked on 180 degrees; orientation-only OOD images mostly flat on
ImageNet-64 and mostly peaked on a wrong angle for CIFAR-100 vs CIFAR-10.  G: the ordering of the datasets by the ratio changes
little with R.  H: whitening recovers part of the flip detection on the hard datasets but damages natural-OOD detection, as the
standardised Euclidean rule did.
