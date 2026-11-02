# ct_bench.py -- throughput (images / s) of Euphonium and of the frozen PROTOCOL_CT configuration on one GPU: same model, batch
#   and precision (DG_FP16 as in the rails runs), 2000 ID test images, two warm-up batches, best of three timed passes over the
#   2000 images. The bank features are computed once beforehand (not timed) and kept on the GPU (fp32, as in ct_extract.py).
#   It also checks that this implementation reproduces the stored content statistic of ct_extract.py on 250 development images.
#   usage: DG_FP16=.. python code/ct_bench.py --tag edmc10 --gpu 0 --choice runs/ct/choice.json
import os, sys, time, json, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgcore import *
from ct_extract_defs import SPEC, SIGS, GROUPS, groups_for, bank_indices

p = argparse.ArgumentParser()
p.add_argument('--tag', required=True); p.add_argument('--gpu', type=int, default=0)
p.add_argument('--choice', required=True); p.add_argument('--n', type=int, default=2000); p.add_argument('--reps', type=int, default=3)
a = p.parse_args()
spec, bankset, bs = SPEC[a.tag]
dev = torch.device(f'cuda:{a.gpu}'); torch.backends.cudnn.benchmark = True
den, net = load_model(spec, dev)
ch = json.load(open(a.choice)); kc, g, v, b, q = ch['content']; ATOMS = [tuple(x) for x in ch['atoms']]; lam = ch['lam']
KK = (1, 10)[q]; SIGNS = (0,) if v == 0 else (0, 1)
hooks = groups_for(net, den, bankset, dev)[GROUPS[g]]; nh = len(hooks)
mods = dict(net.named_modules()); store = {}; active = [False]


def grab(name):
    def f(m, i, o):
        if not active[0]: return
        if isinstance(o, (tuple, list)): o = o[0]
        o = o.float(); store[name] = torch.cat([o.mean((2, 3)), o.std((2, 3), correction=0)], 1)
    return f


for n in hooks: mods[n].register_forward_hook(grab(n))
TFR = ['rot90', 'rot180', 'rot270']


def err(x, y, sig): return (x - y).square().flatten(1).mean(1) / sig ** 2


@torch.no_grad()
def content_pass(x, j):
    eps = shared_noise(x.shape[1:], 0, kc, j, dev); store.clear(); active[0] = True
    y = den(x + SIGS[kc] * eps, SIGS[kc]); active[0] = False
    return {n: store[n] for n in hooks}, err(x, y, SIGS[kc])


def content_vec(P):
    if len(P) == 1: return torch.cat([torch.nn.functional.normalize(P[0][n], dim=1) for n in hooks], 1)
    return torch.cat([torch.nn.functional.normalize((P[0][n] + P[1][n]) / 2, dim=1) for n in hooks], 1)


# bank (not timed)
full = np.load(f'{NPY}/{bankset}.npy', mmap_mode='r'); bidx, small = bank_indices(len(full))
xb = full[bidx[small]] if b == 0 else full[bidx]
Bl = []
with torch.no_grad():
    for i in range(0, len(xb), bs):
        x = to_t(np.ascontiguousarray(xb[i:i + bs]), dev)
        Bl.append(content_vec([content_pass(x, j)[0] for j in SIGNS]).half())
Bm = torch.cat(Bl).float().T.contiguous(); del Bl


@torch.no_grad()
def new_method(x):
    up, P = {}, []
    for j in SIGNS:
        d, e = content_pass(x, j); P.append(d); up[(kc, j)] = e
    K = ((content_vec(P) @ Bm) / nh).topk(KK, dim=1).values[:, -1]
    if not ATOMS: return K
    O = torch.zeros_like(K)
    for T, ka, va in ATOMS:
        f, fi = TF[TFR[T]]; eT = 0; eI = 0
        for j in ((0,) if va == 0 else (0, 1)):
            y = x + SIGS[ka] * shared_noise(x.shape[1:], 0, ka, j, dev)
            if (ka, j) not in up: up[(ka, j)] = err(x, den(y, SIGS[ka]), SIGS[ka])
            eT = eT + err(x, fi(den(f(y), SIGS[ka])), SIGS[ka]); eI = eI + up[(ka, j)]
        O = O + torch.log(eT / eI)
    return K + lam * O          # the z-scaling of the frozen rule is two affine maps (constants from the development images)


@torch.no_grad()
def euphonium(x):
    s, up = 0, {}
    for T, ka in (('rot90', 0), ('rot90', 6), ('rot180', 0)):
        f, fi = TF[T]; eT = 0; eI = 0
        for j in (0, 1):
            y = x + SIGS[ka] * shared_noise(x.shape[1:], 0, ka, j, dev)
            if (ka, j) not in up: up[(ka, j)] = err(x, den(y, SIGS[ka]), SIGS[ka])
            eT = eT + err(x, fi(den(f(y), SIGS[ka])), SIGS[ka]); eI = eI + up[(ka, j)]
        s = s + torch.log(eT / eI)
    return s


sets = json.load(open(f'{ROOT}/runs/ct/sets_{a.tag}.json'))
name, data, idx, pre = sets[0]                                             # the ID development images
X = np.load(f'{NPY}/{data}.npy', mmap_mode='r')[np.asarray(idx[:250])]
with torch.no_grad():
    x = to_t(np.ascontiguousarray(X), dev); Kb = ((content_vec([content_pass(x, j)[0] for j in SIGNS]) @ Bm) / nh).topk(KK, dim=1).values[:, -1]
Ks = np.load(f'{ROOT}/runs/ct/{a.tag}/k{kc}/{name}.npz')['K'][:250, g, v, b, q]
check = float(np.abs(Kb.cpu().numpy() - Ks).max())
xt = np.load(f'{NPY}/{data}.npy', mmap_mode='r')[:a.n]
xs = [to_t(np.ascontiguousarray(xt[i:i + bs]), dev) for i in range(0, len(xt), bs)]


def bench(fn):
    for x in xs[:2]: fn(x)
    torch.cuda.synchronize(); best = 0.0
    for r in range(a.reps):
        t0 = time.time()
        for x in xs: fn(x)
        torch.cuda.synchronize(); best = max(best, len(xt) / (time.time() - t0))
    return best


r_eu = bench(euphonium); r_new = bench(new_method)
res = dict(tag=a.tag, gpu=torch.cuda.get_device_name(a.gpu), fp16=FP16, batch=bs, n=len(xt), bank=int(Bm.shape[1]), dim=int(Bm.shape[0]),
           hooks=hooks, euphonium_img_s=round(r_eu, 1), new_img_s=round(r_new, 1), speedup=round(r_new / r_eu, 2), nfe_new=ch['nfe'], nfe_eu=10,
           check_max_abs_diff_vs_extract=check)
print(json.dumps(res))
os.makedirs(f'{ROOT}/runs/ct/bench', exist_ok=True); json.dump(res, open(f'{ROOT}/runs/ct/bench/{a.tag}.json', 'w'), indent=1)
