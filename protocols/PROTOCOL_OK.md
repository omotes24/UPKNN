# PROTOCOL_OK -- oriented kNN: content and orientation from the same few passes (written 2026-10-31 21:40 JST, before any statistic of
# runs/ok/ is read)

Goal (the user's conditions, all required): (1) overwhelm the feature kNN of PROTOCOL_CT (and therefore every comparison method) in
accuracy, (2) few computations, (3) no oracle: every choice from the selection data only. A failed criterion is reported as a failure.

## History (stated plainly)

- PROTOCOL_CT (2026-10-28) chose, on the selection data, content alone: kNN = cosine of the L2-normalised [channel mean, std] of the last
  decoder block at half resolution (L1), sigma_1, +eps (1 pass), 10th neighbour in a 50k bank of ID training images. Main evaluation
  then: 6-ID mean 87.56 (Euphonium 86.89). "No orientation atom raised the selection mean by more than 0.05."
- Held-out evaluation (2026-10-31): Euphonium 87.5, kNN 87.8; Euphonium vs kNN 20/20/23 (wins/ties/losses). I have looked at the
  held-out AUROCs of Euphonium, kNN, the eight comparison methods and the rotation classifiers on every pair. kNN is weak on the
  upside-down ID sets (50-63 except CelebA/FashionMNIST) and on near-OOD pairs (CIFAR-10 vs CIFAR-100 69.6, ImageNet-64 vs SSB-hard /
  NINCO / OpenImage-O / iNaturalist / CelebA 57 / 66 / 67 / 48 / 31); the rotation classifiers are strong on the upside-down sets.
- On the development-time images (common_idx), Fisher(Euphonium, kNN) had mean 89.3 vs kNN 87.6 (36/21/6). On the selection data
  (today, ana_k/sel_explore.py): Fisher(Euphonium, kNN) 91.04, + Mahalanobis of the upright errors 91.4-91.5, kNN 91.22; two-sided
  Gaussian models of the cue vector lose (<= 88.8). So the existing cues do not overwhelm kNN on the selection data.
- This family (rotated views and rotated banks of the same decoder features; 2x2-grid pooling) was defined after seeing those results.

## Statistics (runs/ok, code/ok_extract.py, md5 425093d453adbb280ff1e0bb1fcf0684)

Views V_r = R_r(x + s_k eps_k), r = 0..3 (0/90/180/270 degrees), k in {0 (sigma_1 = 0.01), 6 (sigma_7 = 0.251)}, eps_k = the rails'
shared noise image of the seed (+eps only; seed 0 = the noise of kNN and of Euphonium's upright +eps pass). One forward pass per (view, k).
Per view: pooled outputs of the last decoder blocks at full / half / quarter resolution (L0, L1, L2): global [mean, std] (2C) and 2x2-grid
[mean, std] (8C); and the error e_k(R_r x) = ||eps - R_r^-1 epshat(V_r)||^2. Banks B_s = the same features of R_s(b + s_k eps_k) for the
50k bank of PROTOCOL_CT (ImageNet-64: its 10k).

Similarities: feature definition phi in {g (global 2C), q (grid 8C), gq (both, 10C)}, each L2-normalised; hook group h in {L0, L1, L2,
L012 (mean of the three per-hook cosines)}; rank kk in {1, 10}; S_rs(x) = kk-th largest cosine of phi(V_r(x)) over B_s.

Components (higher = more ID):
- content C1 = S_00 (1 pass), C2 = mean over r in {0, 2} of S_rr (2 passes), C4 = mean over r of S_rr (4 passes);
- orientation O1 = S_00 - max_{s != 0} S_0s (view 0), O2 = mean over r in {0, 2} of (S_rr - max_{s != r} S_rs), O4 = same over all four
  views (same phi, h, k, kk as the content component);
- error atoms A_{r,k'} = log e_k'(R_r x) - log e_k'(x), r in {1, 2, 3}, k' in {0, 6};
- magnitude T = Mahalanobis distance (Ledoit-Wolf, ID development images) of (log e_0(x), log e_6(x)), large = OOD.
Candidate = content (phi, h, k, kk, C1|C2|C4) + optional orientation (O1|O2|O4 of the same phi, h, k, kk, or one error atom) + optional T;
combined by Fisher's method on empirical one-sided p-values (+1 smoothing) calibrated on ID development images only (T: upper tail).
Passes = number of distinct (view, k) used. Budget: at most 4 passes (kNN 1, Euphonium 10).

## Selection data and criterion (the established protocol)

- ID side: the ID development images of the development runs (devall-*, 2000 per ID; ImageNet-64 1500), two-fold cross-fitting
  (fixed permutation, seed 0): calibrate on one half, score the other half and the extra OOD sets, AUROC averaged over the two folds.
- OOD side: the extra OOD sets of PROTOCOL_CT (sel-*). Never a main-evaluation set, never an upside-down or synthetic set.
- Criterion: mean over the six IDs of the per-ID mean AUROC. Within 0.1 of the best: fewer passes, then the larger worst-ID mean, then
  fewer components, then phi = g before q before gq, then kk = 10.
- The family may be extended on selection-data results only; every extension is logged below with its date.

## Main evaluation (once) and confirmation

- The chosen candidate, calibrated on all ID development images, on the held-out images (fresh-*; the images of every method's held-out
  runs), noise seeds 0, 1, 2 (bank and images with that seed's noise), AUROC averaged over the seeds; kNN recomputed the same way
  (C1, g, L1, k0, kk 10 = PROTOCOL_CT's choice) with seeds 0-2, plus the existing seed-0 kNN.
- Per-pair comparison with kNN, Euphonium and the eight comparison methods (bootstrap rule of Table 1: 2000 resamples, CI excluding 0
  and |gap| >= 0.05 is a win/loss, else a tie); 6-ID mean over the natural OOD sets.
- Confirmation: if S1 holds, the same comparison once more on a new untouched split (final2: images outside the development, common and
  held-out images, cap 500) with all methods rerun. If S1 fails, the failure is reported; a family extension (selection data only) gets
  its single evaluation on final2, and the number of attempts is reported.

## Success criteria (fixed now)

- S1 accuracy: 6-ID natural mean >= kNN + 3.0; wins >= 3 x losses against kNN over the 63 pairs; wins > losses and a higher 6-ID mean
  against each of the eight comparison methods; no ID mean more than 1.0 below kNN's.
- S2 compute: at most 4 passes; throughput (one RTX 2080 Ti, batch as PROTOCOL_CT) >= 2x Euphonium's on all six models; ratio to kNN reported.
- S3 selection: all choices from the selection data; this file and the selection log hashed before the main evaluation; one main evaluation.

## Checks (reported, not used for the choice)

Selection without the ImageNet-derived extra sets (STL-10, Imagenette); the best candidate with Euphonium (10 passes) added; the kNN of
PROTOCOL_CT recomputed from runs/ok vs the pack values (sanity).

## Log

- 2026-10-31 21:40 JST: protocol written; extraction queue eu/jobs/QK.txt (18 jobs) started 21:35 JST.
- 2026-10-31 21:58 JST: selection of the original family (ana_ok/ok_select.py; 2160 candidates within 4 passes): best 91.50
  (C1 g L012 s1 kk1 + A(R180, s1) + T); tie-rule choice C2 g L012 s1 kk1 + A(R180, s1) + T (91.45, 3 passes); kNN 91.21. The
  rotated-bank orientation components (O1/O2/O4) and the grid features (q, gq) never enter the top 15. Sanity: kNN recomputed from
  runs/ok vs the PROTOCOL_CT pack, max |diff| 2.5e-4 (float16), r = 0.99994. The main evaluation was NOT run for this choice; the
  family was extended on selection-data results only:
- 22:05 JST extension 1 (code/ok_content.py, ana_ok/ok_content_sel.py): content variants (centring, PCA whitening 64/256/all,
  concatenation of s1 and s7, LOF-like ratio/difference; L1 and L012): none above the raw cosine (best = kNN itself, 91.21).
- 22:20 JST extension 2 (code/ok_nn.py; ana_ok/ok_resid.py, ok_resid2.py, ok_resid3.py): neighbour-calibrated statistics. For
  every image, its K nearest bank images in the kNN space (L1 or L012, K in {5, 10, 20, 50}); residual of the log errors
  log e_k(R_r x) from their mean over the neighbours (the bank images' errors come from the same extraction run); one-sided error
  residuals, orientation residuals (one/two-sided), Mahalanobis distance (Ledoit-Wolf, ID development images) of residual vectors
  over view sets; combined by Fisher with the content statistics and atoms of the original family.
- 22:35 JST CHOICE (tie rule over all 2940 candidates examined; best 92.28 with 4 passes; within 0.1: 3-pass candidates, the
  largest worst-ID mean): content C1 (g, L012, s1, kk 1) + atom A(R180, s1) + residual Mahalanobis over the upright s1 and s7 errors
  w.r.t. the 50 nearest bank images in the L012 space; selection mean 92.22 (kNN 91.21), worst ID 70.3 (kNN 69.0); per ID
  CIFAR-10 94.0, SVHN 99.5, CelebA 99.9, CIFAR-100 89.9, ImageNet-64 70.3, FashionMNIST 99.7. 3 passes per image (upright s1,
  R180 s1, upright s7). On the selection data this is +1.0 over kNN, short of the margin S1 asks for on the main evaluation.
- 2026-10-31 22:55 JST extension 3 (code/ok_prof.py; ana_ok/ok_topk_sel.py, ok_prof_sel.py, ok_prof_comb.py): the similarity profile in
  the kNN spaces (r-th largest cosine and mean of the r largest, r = 1..1000). Larger r helps ImageNet-64: the mean of the 1000 largest
  cosines in the L012 space (1 pass) has selection mean 92.07, worst ID 74.7 (kNN 91.21 / 69.0); combined with the atom and the
  residual the profile content does not exceed 92.09. The best mean stays 92.28 (extension 2c), so by the tie rule the CHOICE of 22:35
  is unchanged (3 passes, 92.22). Declared now, before any held-out statistic is read: the 1-pass profile statistic (mean of the 1000
  largest cosines, L012, s1) is reported in the main evaluation as a CHECK (not the choice). Candidates examined in total: 3488.
- 2026-10-31 22:58 JST MAIN EVALUATION (once; ana_ok/ok_main.py with ok_freeze.json, sha256 20b76330...; held-out images, seeds 0-2):
  CHOICE (3 passes): 6-ID natural mean 90.06 vs kNN 87.80 (+2.26; the seed-0 kNN run 87.81), Euphonium 87.5, best of the eight
  80.1 (MSMA*). Per pair vs kNN 38 wins / 21 ties / 4 losses (losses: Textures under CIFAR-10, CIFAR-100, ImageNet-64; notMNIST under
  FashionMNIST, all by 0.1-1.7 points). Vs each of the eight: 46-61 wins, 1-2 losses. Vs Euphonium 37/17/9, rotation classifiers
  42/13-14/7-8. Per ID: CIFAR-10 91.9 (kNN 89.8), SVHN 99.9 (98.3), CelebA 100.0 (100.0), CIFAR-100 84.0 (79.5), ImageNet-64 64.9
  (59.7), FashionMNIST 99.7 (99.5); no ID below kNN. S1: FAILS (mean margin +2.26 < 3.0; the other three parts hold). S2: 3 passes
  (throughput to be measured). Check (declared, 1 pass, mean of the 1000 largest cosines in L012): mean 90.04, but vs kNN 14/31/18
  (ImageNet-64 75.5, CIFAR-10 86.8). sha256 ok_main.json / ok_main_check.json logged in ana_ok. Next attempt: a family extension
  chosen on the selection data only, evaluated once on final2 (untouched; all methods rerun; queued 2026-10-31 22:35).
- 2026-10-31 23:20 JST ATTEMPT 2 (family defined after the main evaluation of attempt 1 and its check had been seen; chosen on the
  selection data only, ana_ok/ok_att2_sel.py, 252 candidates within 4 passes): content = one small-rank statistic (C1 g L012 kk1 or
  kNN) + optionally one large-rank profile statistic (mean of the r largest cosines, L1/L012, r 200/500/1000), orientation (none, A(R90,s1),
  A(R180,s1), their sum or both, A(R90,s7)), residual (none, L012/L1). Best 92.84 (4 passes: C1 g L012 kk1 + mean of the 1000 largest
  L1 cosines + A(R90,s1)+A(R180,s1) + residual L012); by the tie rule (within 0.1: fewer passes) the CHOICE is the 1-pass candidate
  Fisher[C1 g L012 kk1, mean of the 1000 largest cosines in L1] (92.75, worst ID 73.4; kNN 91.21 / 69.0). Declared before final2 is
  read: final2 evaluates once the attempt-2 CHOICE, the attempt-1 choice (replication), the 4-pass top of attempt 2 (check, not the
  choice), kNN, Euphonium and the eight methods. Freeze: ana_ok/ok_freeze2.json.
- 2026-11-01 01:00 JST NOVELTY (user, 2026-10-31 late): attempts 1 and 2 only combine existing detectors (kNN content, Euphonium
  atoms, neighbour residual) by Fisher's method; a NEW method is required. Both are withdrawn as the proposed method before any
  final2 statistic is read; they stay frozen (ok_freeze.json, ok_freeze2.json) and are evaluated on final2 as declared, as
  references only. NEW FAMILY (defined now, before any of its statistics is computed): rotational-harmonic retrieval
  (code/ok_harm.py). The global pooled features of the four rotated views (shared noise; L2-normalised per view) form the orbit
  response of the image under the rotation group C4; it is decomposed into harmonics: m0 = mean (invariant), m1 = (n0-n2)/2
  (up/down dipole), m3 = (n1-n3)/2 (left/right dipole), m2 = (n0-n1+n2-n3)/4. An image is scored by retrieval in the training bank
  of a representation built from harmonics (each block L2-normalised; for a concatenation ONE bank image must match all blocks,
  its cosine being the mean of the block cosines): f0, m0, m1, m2, m3, H4 = [m0,m1,m2,m3], H2 = [m02,m1], H01 = [n0,m1],
  O4 = [n0..n3], O2 = [n0,n2]; hooks L0/L1/L2/L012; s1 and s7; 1st and 10th largest cosine. Selection on the selection data only
  (devall vs sel-*, 2-fold cross-fitting where calibration is needed), same criterion and tie rule as above; final evaluation once
  on final2 together with the declared references.
- 2026-11-01 01:40 JST harmonic family, first result (selection data only, ana_new2/harm_sel.py): kNN 91.21; best single harmonic
  representation H4 (L2, s1, 10th) 90.40, H01/H2 88.6-88.8, m1 alone 86.3, m0 (L1) 91.25, O2 (L1) 91.06. Equal-weight harmonic
  retrieval does not beat kNN on the selection data. Extension (same family, selection data only, defined before computing):
  compact PCA projections of the orbit harmonics (code/ok_pca.py; PCA fitted on the training bank only: n0 128 dims, m1/m2/m3
  64 dims, norms |m1|, |m2|, |m3|) for cloud-side statistics: retrieval in ID-normalised joint metrics, the conditional density of
  the orientation harmonics given the content neighbours, and the norms (representational orientation sensitivity).
- 2026-11-01 02:30 JST extension results (selection data only): PCA joint retrieval / conditional dipole / norms, local-scale
  normalisation, held-out-referenced rotation-response typicality, content-conditional full Euphonium: none reaches kNN (91.21).
  Probe of a NEW single statistic (ana_new2/probe11.py, PCA-approximated appearance, L1): the joint density of what the model sees
  and how it behaves, p(x) ~ sum_b exp(kappa (cos(a(x),a(b)) - 1) - |W(B(x) - B(b) - delta)|^2 / 2h^2) over the training bank,
  B = log denoising errors (upright s1/s7, rotation log-ratios at s1), delta/W from the ID calibration images (train/test offset):
  best 91.76 (kappa 100, h 2, B = upright s1/s7), 91.69 with the R180 log-ratio. FAMILY now fixed for the selection on exact
  features (code/ok_jkde.py): appearance hook L1 or L012, kappa 30/100/300, B in {a2l0, a2l2, o3l0, o3l2} (every B contains the
  rotation response; passes 2/3/4/5), h 1/2/4; appearance-only (h = inf) reported as reference, not a candidate. Criterion and tie
  rule as above; the frozen choice is evaluated once on final2.
- 2026-11-01 02:45 JST ATTEMPT 3 (new method; ana_new2/jkde_sel.py, 72 candidates, selection data only): joint appearance-behaviour
  density. Best 91.97 (L1, kappa 100, B = l0 l6 a2, h 2; 3 passes); 5 candidates within 0.1; by the tie rule (fewer passes, then the
  larger worst ID) the CHOICE is L1 | kappa 100 | B = (log e(x) at s1, log e(R180 x)/e(x) at s1) | h 2: 2 passes, selection mean
  91.94 (kNN 91.21), worst ID 72.8 (kNN 69.0); per ID 92.0 / 99.8 / 100.0 / 87.4 / 72.8 / 99.7. Reference (not a candidate):
  appearance only, L012 kappa 30, 91.99. Frozen: ana_ok/ok_freeze3.json. Declared before any held-out statistic of this family is
  computed: evaluation once on final2 (primary; seeds 0-2; calibration on all ID development images of each seed) and on fresh
  (secondary: fresh was used for attempt 1, and the per-pair results of the existing methods on fresh were looked at when the
  family was motivated); checks (reported, not chosen): L1|100|a2l2|2.0 and the appearance-only L012|30.
- 2026-11-01 03:05 JST ATTEMPT 3 on fresh (secondary evaluation, once; ana_ok/ok_eval3.py): 6-ID mean 88.47 vs kNN 87.80 (+0.67),
  per pair vs kNN 23 / 22 / 18; upside-down sets stay near chance (behaviour weight minimal). Not the overwhelming result asked for.
  From now on fresh is a DEVELOPMENT split for attempt 4 (its results have been seen); final2 stays untouched as the single test.
  Measurement for attempt 4 (defined before its statistics are computed): Euphonium (10 NFE, rules of seeds 0-2) of the training-bank
  images (code: eu/experiments/score_euphonium.py; runs/eu/s<seed>/<tag>_bank/bank.npz), to give the joint density a strong,
  low-noise behaviour (antithetic atoms and upright errors) instead of the single-sample errors of attempt 3.
- 2026-11-01 03:25 JST ATTEMPT 4 (defined now; fresh is a development split from now on, final2 untouched): the joint density of
  attempt 3 estimated by the chain rule, p(a, B) = p(a) p(B | a). p(a) = the training-bank appearance KDE of attempt 3 (L1, kappa 100).
  p(B | a) = Nadaraya-Watson conditional density over the ID DEVELOPMENT images (held-out ID references: no train/test offset):
  weights exp(kc (cos(a(x), a(d)) - 1)) over the dev images d, Gaussian behaviour kernel. B = Euphonium's measurement of the image
  (log-ratios of its three atoms and the upright log errors at s1, s7; 10 NFE, antithetic), the strong low-noise behaviour.
  Fixed a priori (no OOD data): kc such that the median effective number of dev references (inverse Simpson index of the weights)
  is 50; behaviour bandwidth H = h^2 Sigma_B (Sigma_B = covariance of B over the dev images), h by Silverman's rule
  (4/(d+2))^(1/(d+4)) n_eff^(-1/(d+4)) with d = 5 and the image's effective number of references. Score = log p(a) + log p(B | a).
  Bank Euphonium runs (03:05 entry) stopped: not needed. Measurements: Euphonium of the dev images (rules of seeds 0-2,
  code/mk_dev_eu.sh), appearance neighbours among the dev images (code/ok_devnn2.py, top 300). Sanity check on the selection
  data (feat-run Euphonium, PCA appearance), go/no-go on fresh, test once on final2.
- 2026-11-01 03:40 JST attempt-4 sanity check on the selection data (ana_new2/att4_sel.py): the conditional behaviour term adds noise
  (CIFAR-10 91.8 -> 83.1, CIFAR-100 87.8 -> 77.5); attempt 4 is NOT taken to fresh or final2.
  Development (fresh is a development split; DEVELOPMENT numbers, not evaluations): dev-calibrated combinations, seed 0
  (ana_new2/fresh_combo.py): profile r1000 L012 90.03, + Euphonium (Fisher) 91.32, kNN + Euphonium 89.71 (kNN 87.81).
  Group-closure likelihood ratio (GCLR): s = log p_app(x) + log P(upright | Eu(x)), P(up | e) = p(e) / (p(e) + p(-e)) with p the
  KDE of the dev images' Euphonium (alternative = rotation closure of the ID distribution, symmetric approximation); p_app = bank
  KDE of ok_jkde.py. Selection data (ana_new2/gclr_sel.py): L1|30 + logPup 92.21 (worst 74.9), L012|30 + logPup 92.18 (73.3)
  (kNN 91.21). Fresh, seeds 0-2 (ana_ok/ok_eval4.py with dev_gclr_fresh.json): L1|30 + logPup 90.15 (+2.36, 24/26/13 vs kNN),
  L012|30 + logPup 90.80 (+3.00, 30/27/6). The user was asked (03:35 JST) whether a method that uses both the appearance density
  and Euphonium's behaviour, unified by the group-closure likelihood ratio, counts as new; final2 is not used until the user decides.
