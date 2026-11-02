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
