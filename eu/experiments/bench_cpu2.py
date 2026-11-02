# bench_cpu2.py <method> [B] -- CPU seconds and network evaluations per image of every detector of the final resume, measured in
#   ONE session on the same machine (2 pinned cores, fp32, the CIFAR-10 network of NVlabs EDM (DDPM++, 56M parameters) with random
#   weights: the compute does not depend on the weights).  Extends experiments/bench_cpu.py with the methods added for the final
#   resume.  Appends one JSON line to runs/bench/bench_cpu2.jsonl.
#   methods: euphonium (10 passes, measured, not extrapolated), diffpath, gepc, msma, eigen (ours: 180), eigpaper (EigenScore in
#   its paper's setting: 10 times x 20 redraws x 3 iterations x 2 x 3 = 3600), nll, ic, ode, kamkari (PF-ODE likelihood + the
#   normal-bundle LID with K = 4D = 12288 scores and the 3072 x 3072 Gram eigendecomposition), lmd (10 inpaintings x 1000 steps
#   + LPIPS), rotcls (two binary ResNet-18 classifiers), rot4 (four-way ResNet-18, four rotations), yoshi (1 pass + features +
#   cosine 10-NN over a 50,000-image bank)
import os, sys, json, time, math, io
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.environ.get('SR_ROOT', os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, f'{ROOT}/ext/edm'); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import core
core.FP16 = False
from training.networks import EDMPrecond
import baselines as BL
from euphonium import euphonium, load_rule
from rotcls_net import build
torch.manual_seed(0)
torch.set_num_threads(int(os.environ.get('BENCH_THREADS', '2')))


class Count:
    def __init__(self, den): self.den, self.n, self.sigma_min, self.sigma_max = den, 0, den.sigma_min, den.sigma_max
    def __call__(self, x, sigma): self.n += x.shape[0]; return self.den(x, sigma)


meth = sys.argv[1]; B = int(sys.argv[2]) if len(sys.argv) > 2 else 4
net = EDMPrecond(img_resolution=32, img_channels=3, label_dim=0, model_type='SongUNet', embedding_type='positional',
                 encoder_type='standard', decoder_type='standard', channel_mult_noise=1, resample_filter=[1, 1],
                 model_channels=128, channel_mult=[2, 2, 2], augment_dim=9, dropout=0.13).eval().requires_grad_(False)
den0 = core.Denoiser(core.edm_precond_fn(net.model, 0.5, fp16=False), 'edmc10'); den = Count(den0)
x = torch.rand(B, 3, 32, 32, generator=torch.Generator().manual_seed(0)) * 2 - 1
SIG20 = np.exp(np.linspace(np.log(0.004), np.log(10.0), 20))
vjp = [0]


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
def lid(x, chunk=256):                                            # kamkari_lid.py for one image at a time
    ib = 0.1 * 0.01 + 0.5 * 19.9 * 0.01 ** 2; al = math.exp(-0.5 * ib); sig = math.sqrt((1 - al ** 2) / al ** 2)
    out = []
    for n in range(x.shape[0]):
        D = x[n].numel(); K = 4 * D; G = torch.zeros(D, D, dtype=torch.float64)
        for c in range(0, K, chunk):
            m = min(chunk, K - c); y = x[n:n + 1] + sig * torch.randn((m,) + tuple(x.shape[1:]))
            S = ((den(y, sig) - y) / (al * sig ** 2)).flatten(1).double(); G += S.T @ S
        out.append(torch.linalg.eigvalsh(G))
    return out


@torch.no_grad()
def lmd(x, steps=1000, reps=10):                                  # lmd.py
    import lpips
    lp = lpips.LPIPS(net='alex', pretrained=True, lpips=True, verbose=False).eval()
    n = x.shape[0]; X = x.repeat(reps, 1, 1, 1)
    def checker(inv):
        m = torch.ones(1, 3, 8, 8)
        if inv: m[:, :, ::2, ::2] = 0.; m[:, :, 1::2, 1::2] = 0.
        else: m[:, :, 1::2, ::2] = 0.; m[:, :, ::2, 1::2] = 0.
        return m.repeat_interleave(4, 2).repeat_interleave(4, 3)
    M = torch.cat([checker(r % 2 == 1).expand(n, -1, -1, -1) for r in range(reps)])
    z = X * M + torch.randn_like(X) * (1 - M); ts = torch.linspace(1.0, 1e-3, steps, dtype=torch.float64); dt = -1.0 / steps
    for i in range(steps):
        t = float(ts[i]); al = math.exp(-0.25 * t * t * 19.9 - 0.05 * t); var = 1 - al * al; s = math.sqrt(var) / al
        sc = (al * den(z / al, s) - z) / var; b = 0.1 + 19.9 * t
        zm = z + (-0.5 * b * z - b * sc) * dt; z = zm + math.sqrt(b * -dt) * torch.randn_like(z)
        z = z * (1 - M) + (al * X + torch.randn_like(X) * math.sqrt(var)) * M; zm = z * (1 - M) + al * X * M
    return lp(X, zm.clamp(-1, 1)).view(reps, n).median(0).values


def run():
    if meth == 'euphonium': return euphonium(den, x, load_rule())
    if meth == 'msma':      return [den(x, float(s)) for s in SIG20]
    if meth == 'diffpath':  return BL.diffpath(den, x)
    if meth == 'gepc':      BL.GEPC_SIG = BL.GEPC_SIG[:2]; return BL.gepc(den, x, torch.Generator().manual_seed(0))
    if meth == 'eigen':     return BL.eigen(den, x, torch.Generator().manual_seed(0), 2, 3, 3)
    if meth == 'eigpaper':
        T = np.arange(100, 551, 50) / 1000.0
        BL.EIG_SIG = np.array([math.sqrt(math.expm1(0.5 * 19.9 * t * t + 0.1 * t)) for t in T])
        return BL.eigen(den, x, torch.Generator().manual_seed(0), 20, 3, 3)
    if meth == 'nll':       return nll(x)
    if meth == 'ic':        return nll(x), codelen(x)
    if meth == 'ode':       return nll(x), nll(x.rot90(1, (2, 3)))
    if meth == 'kamkari':   return nll(x), lid(x)
    if meth == 'lmd':       return lmd(x)
    if meth in ('rotcls', 'rot4'):
        with torch.no_grad():
            if meth == 'rotcls':
                nets = [build(3, 1).eval() for _ in range(2)]; return [n(x) for n in nets]
            n4 = build(3, 4).eval(); return [n4(x.rot90(r, (2, 3))) for r in range(4)]
    if meth == 'yoshi':                                            # tap: last decoder block at 16x16 (l1)
        name = [n for n, _ in net.model.named_modules() if n.startswith('dec.16x16_block')][-1]
        feats = {}
        hk = dict(net.model.named_modules())[name].register_forward_hook(lambda m, i, o: feats.__setitem__('h', o))
        with torch.no_grad():
            den(x + 0.01 * torch.randn_like(x), 0.01); h = feats['h']
            v = torch.nn.functional.normalize(torch.cat([h.mean((2, 3)), h.std((2, 3), correction=0)], 1), dim=1)
            s = (v @ BANK[:, :v.shape[1]].T).topk(10, dim=1).values[:, 9]
        hk.remove(); return s
    raise ValueError(meth)


den0(x[:1], 0.3)
if meth == 'yoshi': BANK = torch.nn.functional.normalize(torch.randn(50000, 512), dim=1)   # precomputed bank features (C = 256 channels -> 512)
den.n = 0
t = time.perf_counter(); run(); el = time.perf_counter() - t
r = dict(method=meth, batch=B, sec_per_img=el / B, nfe_fwd_per_img=den.n / B, vjp_per_img=vjp[0] / B, threads=torch.get_num_threads(),
         cpu=os.popen("lscpu | grep 'Model name' | head -1").read().strip(), when=time.strftime('%F %T'))
if meth == 'ic':
    t = time.perf_counter(); codelen(x); r['codelen_sec_per_img'] = (time.perf_counter() - t) / B
print(json.dumps(r), flush=True)
os.makedirs(f'{ROOT}/runs/bench', exist_ok=True)
with open(f'{ROOT}/runs/bench/bench_cpu2.jsonl', 'a') as f: f.write(json.dumps(r) + '\n')
