# Upright kNN

**Completing nearest-neighbour density on diffusion features with the missing orientation factor for out-of-distribution
detection.** Code, protocols, result files and LaTeX source of the IJCAI-27 submission
([`paper/main.pdf`](paper/main.pdf), supplementary material [`paper/supp.pdf`](paper/supp.pdf)).

Let $\bar p(\mathbf x)=\frac14\sum_{\mathbf G}p(\mathbf G\mathbf x)$ be the density averaged over the four rotations by multiples
of 90°. Under the mixture of rotated in-distribution (ID) images, the posterior that $\mathbf x$ is upright is
$P(\mathrm{up}\mid\mathbf x)=p(\mathbf x)/\{4\bar p(\mathbf x)\}$, so for every image

$$\log p(\mathbf x)=\log\bar p(\mathbf x)+\log P(\mathrm{up}\mid\mathbf x)+\log 4 .$$

Channel statistics of a diffusion denoiser hardly change under rotation, so a kNN density on them estimates mainly the first
term. Upright kNN adds the missing second term, estimated by a small MLP trained on the features of rotated bank images, with
weights fixed by the decomposition:

$$s(\mathbf x)=-d\log r_R(\mathbf x)+\log q_\phi(\mathbf R_0\mid\mathbf x),$$

where $r_R$ is the chordal distance to the $R$-th nearest bank image in the space of pooled decoder statistics (one denoiser
pass at the smallest noise level), $d$ the Levina–Bickel intrinsic dimension of the bank features and $q_\phi$ the calibrated
rotation head. One network evaluation per image, no labels, no OOD data, no tuned weights.

## Results (final evaluation, evaluated once after the protocol was frozen)

Six ID datasets with public (or our) diffusion models, 63 ID–OOD pairs, 500 images per set, three noise seeds
([`paper/gen/data/ours_vs_all_final2.json`](paper/gen/data/ours_vs_all_final2.json)). Mean AUROC (%) over the natural OOD sets:

| | CIFAR-10 | SVHN | CelebA | CIFAR-100 | ImageNet-64 | FashionMNIST | mean | upside-down |
|---|---|---|---|---|---|---|---|---|
| **Upright kNN** | **95.4** | **99.7** | **100.0** | **85.5** | **79.6** | **100.0** | **93.4** | 86.6 |
| best of 11 prior methods (CFS, kNN head) | 84.9 | 92.1 | 99.9 | 72.3 | 63.9 | 99.9 | 85.5 | 70.6 |
| kNN on our features, tuned per ID | 93.7 | 99.4 | 100.0 | 85.4 | 77.0 | 99.9 | 92.6 | 72.7 |

Against each of the 11 prior methods (NLL, IC with PNG / JPEG 2000, typicality test, MSMA\*, DiffPath, GEPC, EigenScore, CFS
with its diagonal and kNN heads, a rotation classifier) Upright kNN wins 41–62 of the 63 pairs (bootstrap, 95% interval and at
least 0.05 points). On OpenOOD v1.5 ImageNet-1K at 64×64 it reaches 70.2 / 79.0 near- / far-OOD AUROC against 60.7 / 75.0 for
the best prior method. Full tables: [`paper/main.pdf`](paper/main.pdf), [`paper/supp.pdf`](paper/supp.pdf).

**Status (2026-11-02).** Runs still in progress and not yet in the paper: ImageNet at 256×256 (ADM, pixels), 512×512
(EDM2-XS, latents) and 256×256 (DiT-XL/2, latents) with every image passed through the same 224-px bottleneck
([`protocols/PROTOCOL_HIRES.md`](protocols/PROTOCOL_HIRES.md), [`PROTOCOL_HIRES_DIT.md`](protocols/PROTOCOL_HIRES_DIT.md));
three further comparison methods (reconstruction, LMD, likelihood + LID) on the six ID datasets; chest X-rays, histology and
generators trained with rotations ([`PROTOCOL_E4.md`](protocols/PROTOCOL_E4.md)); bank size on ImageNet-64
([`PROTOCOL_E5.md`](protocols/PROTOCOL_E5.md)). Every protocol was written before its results were seen. The per-image scores
will be published as a release asset once these runs are finished.

## Layout

| folder | contents |
|---|---|
| [`paper/`](paper/) | LaTeX source of the paper (`main.tex`) and of the supplementary material (`supp.tex`), the IJCAI-27 style files, the generators of every number, table and figure (`gen/`, reading the result files in `gen/data/`), `build.sh`, `build_supp.sh`, the PDFs |
| [`protocols/`](protocols/README.md) | the protocols, each written before the evaluation it governs; `FREEZE_UKNN.md5`: checksums of the frozen protocol and code, recorded before the final evaluation |
| [`code/`](code/README.md) | GPU side, exactly as run: data preparation, feature extraction, the rotation head, the scores, the comparison methods, job lists (`code/jobs/`) |
| `eu/experiments/` | shared GPU utilities of the same code base (model loaders, likelihoods, comparison methods, training of our models, the job runner) |
| `runs/ok/sets_<tier>.json`, `eu/runs/jobs/*_idx/` | the benchmark: the images of every set and split (indices into the arrays built by the preparation scripts) |
| [`analysis/`](analysis/README.md) | CPU side: per-image scores → the result files in `paper/gen/data/` (bootstrap comparisons, FPR95, fusions, ImageNet benchmark, EuroSAT, …) |
| [`submission/`](submission/README.md) | the files uploaded with the submission: paper, technical appendix and the anonymised code archive |
| [`tools/`](tools/) | `check_freeze.py` (verifies the frozen checksums), `make_submission_zip.sh`, the path substitutions applied when the code was copied from the GPU server |
| [`checkpoints/`](checkpoints/README.md), [`data/`](data/README.md), [`environment/`](environment/), [`third_party/`](third_party/fetch.sh) | sources of the models and data sets (not redistributed), package versions, external repositories (fetched by a script) |

## Reproducing

| level | command | needs |
|---|---|---|
| 0 | `bash paper/build.sh && bash paper/build_supp.sh` | Python 3 with numpy / matplotlib, pdfLaTeX + BibTeX (TeX Live) |
| 1 | the scripts of [`analysis/`](analysis/README.md) | the per-image scores (release asset, after the remaining runs) |
| 2 | the GPU pipeline of [`code/README.md`](code/README.md) | GPUs (we used 4× RTX 2080 Ti 11 GB), the data sets, the checkpoints |

The scripts expect the repository at `~/upknn`. When the code was copied from the GPU server, only its root path, its
interpreter path and its host name were replaced ([`tools/subs.py`](tools/subs.py)); `python tools/check_freeze.py` undoes
these substitutions and checks the frozen files against the checksums recorded before the final evaluation.

## Licence

Our code and result files are released under the [MIT licence](LICENSE). The pretrained models, the data sets, the
third-party code that the experiments use (fetched by [`third_party/fetch.sh`](third_party/fetch.sh)) and the IJCAI style
files keep their own licences.
