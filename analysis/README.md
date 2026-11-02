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
| (pending) | `final/hires/ana/hires_ana.py`, `final/e4/ana/e4_ana.py`, `final/e5/ana/e5_ana.py` | resolution-controlled ImageNet; chest X-rays, histology and rotation-trained generators; bank size (written before the results) |
| `bench_cpu_session*.jsonl` | `eu/experiments/bench_cpu2.py` (GPU server, CPU timing), `final/ana/bench_ana.py` | seconds per image on a CPU |
| `ablate2_<tier>.json`, `ofk2_<tier>_s0.json`, `fig1_examples.*`, `dR_s0.json`, `terms_final2_s0.npz` | GPU side: `code/kx_ablate2.py`, `code/kx_ofk2.py`, `code/fig1_examples.py`; `d`, `R`, `T` and the two terms of the seed-0 exports of `code/kx_export.py` | ablations, development evaluation, Figure 1, Figure 2 |

Shared modules: `final/ana/ana_common.py` (tiers, common images, loaders), `final/ana/boot.py` (the bootstrap),
`final/ana_ok/ok_common.py` (kNN profiles), `eu_build/Euphonium/analysis/common.py` and `baselines_eval.py` (the eight
likelihood- and statistic-based comparison methods: scores from the per-image statistics, fitted on ID development images
only), `final/ana/export_baselines.py` (their per-image scores on the evaluation images), `final/ana/mk_inputs_*.py` (the
common evaluation images of each split).
