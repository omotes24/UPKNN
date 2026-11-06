# analysis/ — CPU side

The scripts that turn the per-image scores of the GPU runs into the result files of the paper (`paper/gen/data/`). They keep
the folder layout of the machine on which they ran: the root `$UPKNN_ANALYSIS` (default `~/upknn/analysis`) contains the
scripts below and, once the per-image scores are extracted into it (release asset, published after the remaining runs), their
inputs (`final/data_kx/`, `final/data_ok/`, `final/data/`, `final/eu_stage/`, `eu_build/inputs_full/`, `final/*/res/`).

| result file (`paper/gen/data/`) | script | contents |
|---|---|---|
| `ours_vs_all_final2.json`, `ours_vs_all_fresh.json` | `final/ana_kx/ours_vs_all.py` (`SPLIT=final2` / `fresh`) | AUROC of every row on every pair; bootstrap of Upright kNN against every row (2,000 replicates, rng 20261031; win / loss if the 95% interval of the difference excludes 0 and the difference is at least 0.05 points) |
| `fpr95_all_final2.json` | `final/ana_kx/fpr95_all.py` | FPR at 95% TPR |
| `fusion_final2.json` | `final/ana_kx/fusion_final2.py` | naive fusions ($z$-sum, Fisher) of the kNN term with the rotation classifier and with our orientation term |
| `ok_prof_sel.json` | `final/ana_ok/ok_prof_sel.py` | the tuned kNN: per-ID choice among 80 configurations on the selection data |
| `ood1k_ana.json` | `final/ood1k/ana/ood1k_ana.py` | OpenOOD v1.5 ImageNet-1K at 64×64 (near, far, csID, full spectrum) |
| `in256_ana.json` | `final/in256/ana/in256_ana.py` | ImageNet at 256×256 with ADM, without resolution control |
| `e3_ana.json` | `final/e3/ana/e3_ana.py` | EuroSAT |
| `two_notions.json` | `final/ana_two/two_notions.py` | the two terms per image (final evaluation, seeds 0–2): Spearman correlation, AUROC of each term and of the sum, OOD images flagged by one term only or by both at 95% TPR (Table 1 and the supplement) |
| `two_subspace2/<tier>_s<seed>_<split>.json` | GPU side: `code/two_subspace2.py` (`code/run_ts2.sh`) | geometry of the two notions: rotation displacement vs neighbour distance, discriminant / principal / random directions, kNN restricted to them (Table 1 and the supplement) |
| `qual_sel.json`, `qual_ims.npz` | `final/ana_two/qual/select_qual.py` (fixed rule), the images fetched from the arrays | Figure S1 |
| `csi_ana.json` | `final/csi/ana/csi_ana.py` | CSI against Upright kNN with the bootstrap of `ours_vs_all.py` (written before the scores; `PROTOCOL_CSI.md`) |
| `newrows_final2.json` | `final/ana_kx/newrows_final2.py` | DDPM-OOD, LMD and likelihood + LID (dual-threshold AUROC, the frontier of the public `roc_analysis`) on the final split, with the bootstrap of `ours_vs_all.py` |
| `weight_sweep.json` | `final/ana_kx/weight_sweep.py` | the relative weight $w$ of the two terms on the development and final images |
| `hires_ana.json`, `hires_diag.json` | `final/hires/ana/hires_ana.py`; GPU side `code/hires_diag.py` | resolution-controlled ImageNet (ADM 256, EDM2-XS 512, DiT-XL/2 256; Section 6.3 and the supplement); post-hoc diagnostics (pixel statistics of the bottlenecked images, collapse of DiT's pooled statistics) |
| `e4_ana.json` | `final/e4/ana/e4_ana.py` | chest X-rays, histology and rotation-trained generators (written before the results) |
| `e5b_ana.json` | `final/e5b/ana/e5b_ana.py` (`final/e5/ana/e5_ana.py` for the superseded E5 runs) | bank size on ImageNet-64 |
| `two_controls/`, `two_controls2/` | GPU side: `code/two_controls.py`, `code/two_controls2.py` | controls of the two-notions analysis (`PROTOCOL_E6.md`, `PROTOCOL_E8.md`), read directly by `paper/gen/mk_two.py` and `mk_supp.py` |
| `e7_ana.json` | `final/e7/ana/e7_ana.py` | ADM 256×256 with the same final resampling for every set against the resolution-controlled tier (`PROTOCOL_HIRES2.md`) |
| `bench_cpu_session*.jsonl` | `eu/experiments/bench_cpu2.py` (GPU server, CPU timing), `final/ana/bench_ana.py` | seconds per image on a CPU |
| `ablate2_<tier>.json`, `ofk2_<tier>_s0.json`, `fig1_examples.*`, `dR_s0.json`, `terms_final2_s0.npz` | GPU side: `code/kx_ablate2.py`, `code/kx_ofk2.py`, `code/fig1_examples.py`; `d`, `R`, `T` and the two terms of the seed-0 exports of `code/kx_export.py` | ablations, development evaluation, Figure 1, Figure S2 |

Shared modules: `final/ana/ana_common.py` (tiers, common images, loaders), `final/ana/boot.py` (the bootstrap),
`final/ana_ok/ok_common.py` (kNN profiles), `eu_build/Euphonium/analysis/common.py` and `baselines_eval.py` (the eight
likelihood- and statistic-based comparison methods: scores from the per-image statistics, fitted on ID development images
only), `final/ana/export_baselines.py` (their per-image scores on the evaluation images), `final/ana/mk_inputs_*.py` (the
common evaluation images of each split).
