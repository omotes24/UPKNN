# code/ — GPU side

The scripts as they were run on the GPU server (4× RTX 2080 Ti), with only the root path and the interpreter path renamed
([`../tools/subs.py`](../tools/subs.py)). They are run from the repository root (`~/upknn`), read `data/npy/<set>.npy` (built by
the preparation scripts, see [`../data/README.md`](../data/README.md)), `ckpt/<checkpoint>` and `runs/models/` (our models), and
write under `runs/`. The exact commands are in [`jobs/`](jobs/) (queues consumed by `eu/experiments/runq2.py`, one worker per GPU;
`{gpu}` is filled in by the worker).

Names: a *tier* is an ID dataset with its diffusion model (`edmc10` CIFAR-10 / EDM, `l6svhn` SVHN / VP-SDE, `dpceleba` CelebA /
iDDPM, `edmc100` CIFAR-100 / DDPM++, `in64` ImageNet-64 / EDM2-XS, `fmnist` FashionMNIST / our U-Net; `in256`, `in256c`, `e512`,
`dit256`, `eurosat`, `chest`, `path`, `c10*`, `in64h*` for the further experiments). Set names carry the split as a prefix:
`devall-` / `sel-` (selection data), `fresh-` (development evaluation images), `final2-` (final evaluation images, evaluated once),
`ood1k-` (OpenOOD v1.5 ImageNet-1K), `q256-` / `q512-` / `qdit-` (resolution-controlled ImageNet), `e3-` / `e4-` (further tiers).
The image lists are `runs/ok/sets_<tier>.json` (`[name, data array, indices, transform]`).

## Pipeline of Upright kNN

| step | scripts | output |
|---|---|---|
| models, hooks, denoiser API | `dgcore.py` (data, model loading, transforms), `ct_extract_defs.py` (`SPEC`: model and bank of every tier; hook groups: the last decoder block at full, 1/2 and 1/4 resolution), `admwrap.py` (ADM 256), `ditwrap.py` (DiT-XL/2 as an unconditional denoiser), `hfwrap.py`, `dgnet.py` (our U-Net) | — |
| features of the sets | `ok_extract.py` (pooled statistics of the three blocks at $\sigma_1$: channel mean / std over the image and over a 2×2 grid; shared noise image per seed), `ok_extract_v0.py` (view 0 only) | `runs/ok/s<seed>/<tier>/k0/<set>__L{0,1,2}.npy` |
| features of the rotated bank | `kx_extract_rot.py` (the four rotations of the bank images; **the noise is not rotated**), `assemble_rotb.py` | `runs/kx/rotb/s<seed>/<tier>/k0/bank__L*.npy` |
| orientation head | `kx_rothead2.py` (MLP D–512–4 on standardised gq features of the rotated bank views; 5,000 bank images held out; temperature by held-out NLL), `kx_rothead2_lm.py` (same, bank mean / std in chunks), `kx_applyhead.py` (apply a trained head to new sets) | `runs/kx/rothead/s<seed>/<tier>_k0_gqR/` |
| score | `kx_export.py` (frozen), `kx_export_x.py` (further tiers): $s=-d\log r_R+\log q_\phi(\mathbf R_0\mid\mathbf x)$, $d$ by Levina–Bickel on 3,000 bank points | `runs/kx/export/<tier>_s<seed>_<split>.npz` (`score`, `content`, `pose`, `idx` per set) |
| the final evaluation | `run_final2_uknn.sh` (frozen; [`../protocols/PROTOCOL_UKNN.md`](../protocols/PROTOCOL_UKNN.md)) | |

Selection of the configuration (selection data only; [`PROTOCOL_OK.md`](../protocols/PROTOCOL_OK.md), [`PROTOCOL_UKNN.md`](../protocols/PROTOCOL_UKNN.md)):
`ok_sims.py`, `ok_nn.py`, `ok_prof.py` (similarity profiles of the kNN family), `kx_explore.py`, `kx_ofk2.py` (development
evaluation of the score; choice of block set and $R$), `kx_rothead.py` / `kx_rothead3.py` / `kx_rh2scores*.py` / `kx_tta.py`
(head variants), `kx_ablate.py`, `kx_ablate2.py` (ablations on the development images and the selection data), `kx_perpair.py`,
`kx_probe.py` (diagnostic only). The other `ok_*.py` files are the attempts of the same protocol that were not chosen.

## Comparison methods (same diffusion models)

| method | scripts |
|---|---|
| kNN (basic: half-resolution block, 10th neighbour) | `knn_basic.py`, `ood1k_knn.py`, `in256_knn.py`, `e3_knn.py`, `e4_knn.py`; the tuned kNN is a profile of `ok_prof.py` chosen on the selection data |
| NLL, IC (PNG / JPEG 2000), typicality test | `eu/experiments/nll.py` (probability-flow ODE), `eu/experiments/complexity.py` (code lengths) |
| MSMA\* | `extract.py` (score norms at 20 noise levels) |
| DiffPath, GEPC, EigenScore | `eu/experiments/baselines.py` (re-implementations on the denoiser API), `eu/experiments/eig_paper.py` |
| CFS (diagonal and kNN heads) | `cfs_extract.py` / `cfs_extract_x.py` (port of the official hook policy and descriptors), `cfs_score.py` |
| rotation classifier (ResNet-18) | `eu/experiments/rotcls.py`, `rotcls_net.py`, `rotcls_eval.py` |
| reconstruction (Graham et al.), LMD, likelihood + LID (Kamkari et al.) | `eu/experiments/graham.py`, `lmd.py`, `kamkari_lid.py`, `lpca_dim.py` |
| timing | `eu/experiments/bench_gpu.py`, `bench_cpu2.py`, `ok_bench.py`, `ct_bench.py` |

## Further experiments

| experiment | scripts |
|---|---|
| OpenOOD v1.5 ImageNet-1K at 64×64 | `prep_ood1k.py`, `run_ood1k_main.sh`, `mk_ood1k_bl.py`, `run_ood1k_bl.sh`, `mk_rotcls_idx.py` (`vl_*.py`: a detector of the development phase that is not reported) |
| ImageNet 256×256 with ADM (uncontrolled) | `prep_hr.py`, `prep_hr_cs.py`, `mk_sets_in256.py`, `run_in256.py`, `run_in256.sh`, `run_in256_g3.sh`, `run_adm.py` |
| resolution-controlled ImageNet (ADM 256, EDM2-XS 512, DiT-XL/2 256) | `prep_c224.py`, `mk_sets_hires.py`, `enc_e512.py`, `enc_dit.py`, `run_hires.py`, `hires_seed.sh`, `hires_cfs.sh`, `mk_q_hires.py`, `patch_hires.py`, `mk_q_6id.py` |
| EuroSAT (no canonical orientation) | `prep_eurosat.py`, `mk_sets_eurosat.py`, `run_e3.py`, `run_e3.sh`, `run_e3s.sh` |
| chest X-rays, histology, generators trained with rotations | `prep_newid.py`, `mk_sets_e4.py`, `run_e4.py`, `e4_resume.py`, `e4_tier.sh`, `eu/experiments/train.py` |
| bank size on ImageNet-64 | `mk_e5.py`, `run_e5.py`, `e5_export.py`, `e5_tier.sh` |

## Data preparation

`prep_data.py`, `prep_gray.py`, `prep_in64.py`, `eu/experiments/prep_in64_hf.py`, `prep_newood.py`, `prep_newood3.py`,
`prep_eurosat.py`, `prep_newid.py`, `prep_ood1k.py`, `prep_hr.py`, `prep_hr_cs.py`, `prep_c224.py`; set lists:
`mk_ok_sets.py`, `mk_ct_sets.py`, `mk_sets_*.py`, `mk_e5.py`. See [`../data/README.md`](../data/README.md).
