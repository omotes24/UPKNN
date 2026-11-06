# protocols/

Each protocol was written, and copied unchanged here, before the evaluation it governs was run.

| file | governs |
|---|---|
| [`PROTOCOL_CT.md`](PROTOCOL_CT.md) | the kNN rule on pooled diffusion features (the "basic" kNN of the paper): features, selection data, frozen configuration |
| [`PROTOCOL_OK.md`](PROTOCOL_OK.md) | the search over kNN extensions on the selection data (the "tuned" kNN and the content term), with its dated extensions |
| [`PROTOCOL_UKNN.md`](PROTOCOL_UKNN.md) | **Upright kNN**: the frozen configuration and the single final evaluation (`final2`) |
| [`FREEZE_UKNN.md5`](FREEZE_UKNN.md5) | checksums of `PROTOCOL_UKNN.md` and of the frozen code, recorded before the final evaluation (`python tools/check_freeze.py`) |
| [`PROTOCOL_IN256.md`](PROTOCOL_IN256.md) | ImageNet at 256×256 with ADM (ImageNet-64 configuration unchanged) |
| [`PROTOCOL_E3.md`](PROTOCOL_E3.md) | EuroSAT, an ID dataset without a canonical orientation ($R$ fixed in advance) |
| [`PROTOCOL_HIRES.md`](PROTOCOL_HIRES.md), [`PROTOCOL_HIRES_DIT.md`](PROTOCOL_HIRES_DIT.md) | resolution-controlled ImageNet at 256×256 (ADM), 512×512 (EDM2-XS) and 256×256 (DiT-XL/2) |
| [`PROTOCOL_E4.md`](PROTOCOL_E4.md) | chest X-rays, histology and generators trained with rotations |
| [`PROTOCOL_E5.md`](PROTOCOL_E5.md) | bank size on ImageNet-64 (an analysis after the frozen evaluation) |
| [`PROTOCOL_CSI.md`](PROTOCOL_CSI.md) | CSI (official code, 1,000 epochs) as a comparison method on the five 32×32 ID datasets |
| [`PROTOCOL_E5B.md`](PROTOCOL_E5B.md) | bank size on ImageNet-64 with the benchmark preprocessing (25,000 and 50,000 random training images; supersedes the E5 runs, whose bank came from a differently preprocessed source) |
| [`PROTOCOL_E6.md`](PROTOCOL_E6.md) | controls of the two-notions analysis: kNN in the full head space, the estimator-free mean-3 subspace, the head on $\mathbf g$ only, a head on pixel statistics, the sensitivity of $d$ |
| [`PROTOCOL_HIRES2.md`](PROTOCOL_HIRES2.md) | ImageNet at 256×256 (ADM) with the same final resampling for every set (every image through 112 px) |
| [`PROTOCOL_E8.md`](PROTOCOL_E8.md) | further controls: the shape of the head's posterior by decision region, the displacement ratio at fixed neighbour ranks, a whitened (Mahalanobis) kNN |
