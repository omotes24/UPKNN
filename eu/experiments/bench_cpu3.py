# bench_cpu3.py -- the CPU column of Table 2, measured in ONE session (same machine, same process, 2 threads, fp32, the CIFAR-10
#   network of NVlabs EDM (DDPM++, 56M parameters) with random weights: the compute does not depend on the weights):
#   the multi-pass methods once each (as bench_cpu2.py), and the single-pass methods INTERLEAVED (round robin, `reps` repetitions,
#   median), so that run-to-run drift does not land on one method.  Replaces bench_cpu_session.jsonl + bench_cpu_session2.jsonl,
#   whose single-pass rows were measured in separate runs and whose CFS kNN-head row used torch.cdist, which recomputes the bank
#   norms at every call (kept here as cfs_knn_cdist for the record; the score is identical).
#   usage: python bench_cpu3.py [B] [reps]      -> runs/bench/bench_cpu3.jsonl (one line per method; the single-pass lines carry
#          the medians and all repetitions)       methods: nll ic msma diffpath gepc eigen | knn ours cfs cfs_knn cfs_knn_cdist rot4
import os, sys, json, time, math, io
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.environ.get('SR_ROOT', os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, f'{ROOT}/ext/edm'); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import core
core.FP16 = False
from training.networks import EDMPrecond
import baselines as BL
from rotcls_net import build
torch.manual_seed(0)
torch.set_num_threads(int(os.environ.get('BENCH_THREADS', '2')))
B = int(sys.argv[1]) if len(sys.argv) > 1 else 4; REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 15
NBANK = 50000                                                      # CIFAR-10 bank (training images)


class Count:
    def __init__(self, den): self.den, self.n, self.sigma_min, self.sigma_max = den, 0, den.sigma_min, den.sigma_max
    def __call__(self, x, sigma): self.n += x.shape[0]; return self.den(x, sigma)


net = EDMPrecond(img_resolution=32, img_channels=3, label_dim=0, model_type='SongUNet', embedding_type='positional',
                 encoder_type='standard', decoder_type='standard', channel_mult_noise=1, resample_filter=[1, 1],
                 model_channels=128, channel_mult=[2, 2, 2], augment_dim=9, dropout=0.13).eval().requires_grad_(False)
den0 = core.Denoiser(core.edm_precond_fn(net.model, 0.5, fp16=False), 'edmc10'); den = Count(den0)
x = torch.rand(B, 3, 32, 32, generator=torch.Generator().manual_seed(0)) * 2 - 1
SIG20 = np.exp(np.linspace(np.log(0.004), np.log(10.0), 20))
vjp = [0]
MODS = dict(net.model.named_modules())
def last_block(res): return [n for n in MODS if n.startswith(f'dec.{res}x{res}_block')][-1]
TAPS = {'L0': last_block(32), 'L1': last_block(16), 'L2': last_block(8),            # the kNN term and the head (kx_export.py)
        'cfs_enc': 'enc.16x16_block0', 'cfs_dec': 'dec.16x16_block4'}               # CFS, CIFAR-10 (official hook policy, cfs_extract.py)
C = 256                                                                            # channels of every tapped block (channel_mult 2 x 128)
feats = {}
def tap(name):
    return MODS[name].register_forward_hook(lambda m, i, o: feats.__setitem__(name, o[0] if isinstance(o, (tuple, list)) else o))
g = torch.Generator().manual_seed(1)
BANK1 = torch.nn.functional.normalize(torch.randn(NBANK, 2 * C, generator=g), dim=1)          # kNN (basic): L1, [mean, std], unit length
BANK3 = torch.nn.functional.normalize(torch.randn(NBANK, 6 * C, generator=g), dim=1)          # Upright kNN: L0+L1+L2
BANKC = torch.randn(NBANK, 4 * C, generator=g)                                                # CFS kNN head: standardised descriptors
BANKC_SQ = (BANKC * BANKC).sum(1)                                                              # their squared norms, precomputed once
HMU, HSD = torch.randn(30 * C, generator=g), torch.rand(30 * C, generator=g) + 0.5           # standardisation of the head input (3 x 10C)
HEAD = torch.nn.Sequential(torch.nn.Linear(30 * C, 512), torch.nn.ReLU(), torch.nn.Linear(512, 4)).eval().requires_grad_(False)
CMU, CSD = torch.randn(4 * C, generator=g), torch.rand(4 * C, generator=g) + 0.5             # CFS: bank mean / std per slot
D_INT, R_NN, TEMP = 19.8, 2, 1.3                                                              # d, R, T of the frozen CIFAR-10 configuration
R4 = build(3, 4).eval().requires_grad_(False)                                                 # four-way rotation classifier (ResNet-18)


def gstat(h): return torch.cat([h.mean((2, 3)), h.std((2, 3), correction=0)], 1)              # [mean C, std C]
def gqstat(h):                                                                                # + 2x2 grid means and stds (kx_extract_rot.py)
    q = h.unfold(2, h.shape[2] // 2, h.shape[2] // 2).unfold(3, h.shape[3] // 2, h.shape[3] // 2)   # [B, C, 2, 2, h/2, w/2]
    return torch.cat([gstat(h), q.mean((4, 5)).flatten(1), q.std((4, 5), correction=0).flatten(1)], 1)


def nll(x, steps=64, smin=0.002, smax=80.0):                      # experiments/nll.py, grid='log', one Rademacher probe
    sig = torch.exp(torch.linspace(math.log(smin), math.log(smax), steps + 1, dtype=torch.float64))
    v = (torch.randint(0, 2, x.shape[1:], generator=torch.Generator().manual_seed(1234)).float() * 2 - 1)[None].expand_as(x)
    def f_and_div(x, s):
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            f = x - den(x, torch.full([x.shape[0]], float(s)))
            gr, = torch.autograd.grad((f * v).sum(), x); vjp[0] += x.shape[0]
        return f.detach(), (gr * v).flatten(1).sum(1).detach()
    logp = torch.zeros(x.shape[0], dtype=torch.float64)
    for k in range(steps):
        s0, s1 = float(sig[k]), float(sig[k + 1]); h = math.log(s1 / s0)
        f0, d0 = f_and_div(x, s0); f1, d1 = f_and_div(x + h * f0, s1)
        x = x + h * 0.5 * (f0 + f1); logp += h * 0.5 * (d0 + d1).double()
    dim = float(np.prod(x.shape[1:]))
    return -0.5 * (x.double() / smax).square().flatten(1).sum(1) - dim * math.log(smax) - 0.5 * dim * math.log(2 * math.pi) + logp


def codelen(x):
    from PIL import Image
    u = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).numpy()
    out = []
    for im in u:
        for fmt, kw in (('PNG', dict(optimize=True)), ('JPEG2000', dict(irreversible=False))):
            b = io.BytesIO(); Image.fromarray(im).save(b, format=fmt, **kw); out.append(8 * b.tell())
    return out


@torch.no_grad()
def one_pass(names, sigma):
    """one denoiser pass on x + sigma * eps with the named blocks tapped; returns their outputs"""
    hks = [tap(n) for n in names]; feats.clear()
    den(x + sigma * torch.randn(x.shape, generator=torch.Generator().manual_seed(2)), sigma)
    for h in hks: h.remove()
    return [feats[n] for n in names]


@torch.no_grad()
def single(meth):
    if meth == 'knn':                                              # kNN (basic): L1, cosine, 10th neighbour
        h, = one_pass([TAPS['L1']], 0.01); v = torch.nn.functional.normalize(gstat(h), dim=1)
        return (v @ BANK1.T).topk(10, dim=1).values[:, 9]
    if meth == 'ours':                                             # Upright kNN: kNN term (L0+L1+L2, R-th cosine) + orientation term (head on g, q)
        hs = one_pass([TAPS['L0'], TAPS['L1'], TAPS['L2']], 0.01)
        z = torch.cat([torch.nn.functional.normalize(gstat(h), dim=1) for h in hs], 1) / math.sqrt(3)
        c = (z @ BANK3.T).topk(R_NN, dim=1).values[:, R_NN - 1].clamp(max=1 - 1e-6)
        content = -D_INT * 0.5 * torch.log(2 * (1 - c))
        u = (torch.cat([gqstat(h) for h in hs], 1) - HMU) / HSD
        pose = torch.log_softmax(HEAD(u) / TEMP, dim=1)[:, 0]
        return content + pose
    if meth in ('cfs', 'cfs_knn', 'cfs_knn_cdist'):                # CFS: [mean, sqrt(var)] of one encoder and one decoder block
        hs = one_pass([TAPS['cfs_enc'], TAPS['cfs_dec']], 0.0302)
        z = torch.cat([torch.cat([h.mean((2, 3)), torch.sqrt(h.var((2, 3), unbiased=False).clamp(min=1e-6))], 1) for h in hs], 1)
        if meth == 'cfs':                                          # diagonal head: mean standardised square per slot, mean of the slots
            s = ((z - CMU) ** 2 / CSD ** 2); return torch.stack([s[:, :2 * C].mean(1), s[:, 2 * C:].mean(1)], 1).mean(1)
        zs = (z - CMU) / CSD
        if meth == 'cfs_knn':                                      # kNN head: mean Euclidean distance to the 10 nearest bank descriptors
            d2 = (zs * zs).sum(1, keepdim=True) + BANKC_SQ[None] - 2 * zs @ BANKC.T       # bank norms precomputed
            return d2.clamp(min=0).topk(10, dim=1, largest=False).values.sqrt().mean(1)
        return torch.cdist(zs, BANKC).topk(10, dim=1, largest=False).values.mean(1)       # as cfs_score.py (bench_cpu_session2)
    if meth == 'rot4':                                             # rotation classifier: four rotated views through a ResNet-18
        return [R4(x.rot90(r, (2, 3))) for r in range(4)]
    raise ValueError(meth)


def multi(meth):
    if meth == 'msma':      return [den(x, float(s)) for s in SIG20]
    if meth == 'diffpath':  return BL.diffpath(den, x)
    if meth == 'gepc':      BL.GEPC_SIG = BL.GEPC_SIG[:2]; return BL.gepc(den, x, torch.Generator().manual_seed(0))
    if meth == 'eigen':     return BL.eigen(den, x, torch.Generator().manual_seed(0), 2, 3, 3)
    if meth == 'nll':       return nll(x)
    if meth == 'ic':        return nll(x), codelen(x)
    raise ValueError(meth)


CPU = os.popen("lscpu | grep 'Model name' | head -1").read().strip()
os.makedirs(f'{ROOT}/runs/bench', exist_ok=True); OUTF = f'{ROOT}/runs/bench/bench_cpu3.jsonl'
def record(r):
    r.update(threads=torch.get_num_threads(), cpu=CPU, torch=torch.__version__, when=time.strftime('%F %T')); print(json.dumps(r), flush=True)
    with open(OUTF, 'a') as f: f.write(json.dumps(r) + '\n')


den0(x[:1], 0.3)                                                   # warm-up
with torch.no_grad():
    for m in ('knn', 'ours', 'cfs', 'cfs_knn', 'cfs_knn_cdist', 'rot4'): single(m)
SINGLE = ['knn', 'ours', 'cfs', 'cfs_knn', 'cfs_knn_cdist', 'rot4']
T = {m: [] for m in SINGLE}
for rep in range(REPS):                                            # interleaved: every method once per round
    for m in SINGLE:
        den.n = 0; t = time.perf_counter(); single(m); T[m].append((time.perf_counter() - t) / B)
for m in SINGLE:
    den.n = 0; single(m)
    record(dict(method=m, batch=B, sec_per_img=float(np.median(T[m])), reps=REPS, sec_per_img_all=[round(v, 4) for v in T[m]],
                nfe_fwd_per_img=den.n / B, vjp_per_img=0.0, interleaved=True))
for m in ('msma', 'diffpath', 'gepc', 'eigen', 'nll', 'ic'):     # multi-pass methods, once each (as bench_cpu2.py)
    den.n = 0; vjp[0] = 0; t = time.perf_counter(); multi(m); el = time.perf_counter() - t
    r = dict(method=m, batch=B, sec_per_img=el / B, nfe_fwd_per_img=den.n / B, vjp_per_img=vjp[0] / B, interleaved=False)
    if m == 'ic':
        t = time.perf_counter(); codelen(x); r['codelen_sec_per_img'] = (time.perf_counter() - t) / B
    record(r)
