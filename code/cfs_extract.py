# cfs_extract.py -- CFS (Rouzoumka et al., "Backbone-Equated Diffusion OOD via Sparse Internal Snapshots", NeurIPS 2026; arXiv 2605.11014) as a
#   comparison method on our six ID backbones, following the paper's main operating point CFS(1x2) (Table 22) and the official code
#   (cfs/methods/cfs.py of github.com/RouzAY/cfs-diffusion-ood):
#     - one canonical level lambda = 5 (logSNR); in the EDM coordinates of our denoiser wrappers x_lambda = x0 + sigma eps with
#       sigma = exp(-lambda / 2) = 0.0821 and a FRESH eps ~ N(0, I) per image (mc_fit = mc_test = 1);
#     - hook policy sparse_ed_id: candidate modules = the U-Net blocks of the encoder and of the decoder (EDM: *.enc/dec.<res>_block<i>;
#       improved-diffusion: input_blocks.<i>.0 / output_blocks.<i>.0; diffusers-style backbones, which the official code does not cover:
#       down_blocks.<i>.resnets.<j> / up_blocks.<i>.resnets.<j>); the official structural order (encoder: deepest non-bottom resolution
#       first, decoder: highest non-top resolution first, later blocks first), the first 4 of each region ranked by the ID-only proxy
#       with one probe batch of 32 ID training images (the official code's between-image variance of the pooled descriptor, since the
#       within term vanishes for one probe batch); 1 encoder hook + 1 decoder hook;
#     - descriptor per hook: [channel-wise spatial mean, channel-wise spatial std] (Eq. 6).
#   Stores the descriptors of the bank (the ID training images of our kNN bank) and of the sets (fresh-*, final2-*); the heads are fitted
#   and applied by cfs_score.py.  Output: runs/cfs/s<seed>/<tag>/<name>__z.npy [N, D_enc + D_dec] float32, <name>__idx.npy, meta.json
#   usage: DG_FP16=0 python code/cfs_extract.py --tag edmc10 --gpu 0 --seed 0 [--prefix fresh-,final2-] [--probe_only]
import os, sys, re, json, time, zlib, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgcore import ROOT, NPY, load_model, to_t, TF
from ct_extract_defs import SPEC, bank_indices

p = argparse.ArgumentParser()
p.add_argument('--tag', required=True); p.add_argument('--gpu', type=int, default=0); p.add_argument('--seed', type=int, default=0)
p.add_argument('--lam', type=float, default=5.0); p.add_argument('--prefix', default='fresh-,final2-')
p.add_argument('--probe_only', action='store_true'); p.add_argument('--out', default='runs/cfs')
a = p.parse_args()
spec, bankset, bs = SPEC[a.tag]
dev = torch.device(f'cuda:{a.gpu}'); torch.backends.cudnn.benchmark = True
den, net = load_model(spec, dev)
SIG = float(np.exp(-a.lam / 2)); EPS = 1e-6
names = [n for n, _ in net.named_modules()]
FAM, CAND = None, []
for fam, rx in (('edm', r'(^|\.)(enc|dec)\.\d+x\d+_block\d+$'), ('improved', r'^(.*\.)?(input_blocks|output_blocks)\.\d+\.0$'),
                ('diffusers', r'^(.*\.)?(down_blocks|up_blocks)\.\d+\.resnets\.\d+$')):
    CAND = [n for n in names if re.search(rx, n)]
    if CAND: FAM = fam; break
assert CAND, 'no candidate modules'
def region(n):
    if FAM == 'edm': return 'enc' if re.search(r'(^|\.)enc\.', n) else 'dec'
    if FAM == 'improved': return 'enc' if 'input_blocks' in n else 'dec'
    return 'enc' if 'down_blocks' in n else 'dec'
mods = dict(net.named_modules()); store = {}
def grab(n):
    def f(m, i, o):
        if isinstance(o, (tuple, list)): o = o[0]
        if torch.is_tensor(o) and o.ndim == 4: store[n] = o
    return f
def pool(o):                                                       # Eq. (6): [channel mean, channel std] (biased variance, floor 1e-6)
    o = o.float(); return torch.cat([o.mean((2, 3)), torch.sqrt(torch.clamp(o.var((2, 3), unbiased=False), min=EPS))], 1)
bank_all = np.load(f'{NPY}/{bankset}.npy', mmap_mode='r'); bidx, _ = bank_indices(len(bank_all))
g = torch.Generator(device=dev); g.manual_seed(1000003 * a.seed + 17)
probe_i = np.sort(np.random.RandomState(a.seed).choice(len(bidx), 32, replace=False))
xp = to_t(np.ascontiguousarray(bank_all[bidx[probe_i]]), dev)
@torch.no_grad()
def forward_capture(x, hooknames, gen):
    hs = [mods[n].register_forward_hook(grab(n)) for n in hooknames]; store.clear()
    try:
        eps = torch.randn(x.shape, generator=gen, device=dev); den(x + SIG * eps, SIG)
    finally:
        for h in hs: h.remove()
    return {n: store[n] for n in hooknames if n in store}
# shapes of the candidates (one forward pass)
out = forward_capture(xp[:2], CAND, g); cand = [n for n in CAND if n in out]; shp = {n: tuple(out[n].shape) for n in cand}
def order(region_name):                                            # the official _region_structural_order
    idx = [n for n in cand if region(n) == region_name]
    if not idx: return []
    res = lambda n: shp[n][2]
    if region_name == 'enc':
        pref = [n for n in idx if res(n) > min(res(m) for m in idx)] or list(idx); levels = sorted({res(n) for n in pref})
    else:
        pref = [n for n in idx if res(n) < max(res(m) for m in idx)] or list(idx); levels = sorted({res(n) for n in pref}, reverse=True)
    o, used = [], set()
    for rr in levels:
        for n in reversed([n for n in pref if res(n) == rr]):
            if n not in used: o.append(n); used.add(n)
    return o + [n for n in idx if n not in used]
@torch.no_grad()
def proxy(n):                                                      # id_probe_batches = 1: between-image variance of the pooled slot
    z = pool(forward_capture(xp, [n], g)[n]); return float(z.var(0, unbiased=False).mean())
SEL, RANK = [], {}
for rg in ('enc', 'dec'):
    short = order(rg)[:4]; sc = {n: proxy(n) for n in short}; RANK[rg] = sorted(sc.items(), key=lambda t: -t[1]); SEL.append(RANK[rg][0][0])
meta = dict(tag=a.tag, seed=a.seed, lam=a.lam, sigma=SIG, family=FAM, n_candidates=len(cand), hooks=SEL,
            shapes={n: shp[n] for n in SEL}, ranking={k: [[n, s, shp[n]] for n, s in v] for k, v in RANK.items()}, fp16=os.environ.get('DG_FP16', '1'))
print(json.dumps(meta), flush=True)
if a.probe_only: sys.exit(0)
od = f'{ROOT}/{a.out}/s{a.seed}/{a.tag}'; os.makedirs(od, exist_ok=True); json.dump(meta, open(f'{od}/meta.json', 'w'), indent=1)
@torch.no_grad()
def run(xu, idx, pre, name):
    if os.path.exists(f'{od}/{name}__idx.npy'): print('exists', name, flush=True); return
    gen = torch.Generator(device=dev); gen.manual_seed(1000003 * a.seed + zlib.crc32(name.encode()))   # fresh noise per image
    Z = []
    for i in range(0, len(xu), bs):
        x = to_t(np.ascontiguousarray(xu[i:i + bs]), dev)
        if pre != 'none': x = TF[pre][0](x)
        o = forward_capture(x, SEL, gen); Z.append(torch.cat([pool(o[n]) for n in SEL], 1).cpu().numpy())
    np.save(f'{od}/{name}__z.npy', np.concatenate(Z).astype(np.float32)); np.save(f'{od}/{name}__idx.npy', np.asarray(idx, np.int64))
t0 = time.time()
run(bank_all[bidx], bidx, 'none', 'bank'); print(a.tag, 'bank', len(bidx), f'{time.time() - t0:.0f}s', flush=True)
for name, data, idx, pre in json.load(open(f'{ROOT}/runs/ok/sets_{a.tag}.json')):
    if not name.startswith(tuple(a.prefix.split(','))): continue
    x = np.load(f'{NPY}/{data}.npy', mmap_mode='r')[np.asarray(idx, dtype=np.int64)]; run(x, idx, pre, name)
print('CFS_EXTRACT_DONE', a.tag, a.seed, f'{time.time() - t0:.0f}s', flush=True)
