# ok_bench.py -- throughput (images / s) on one GPU of Euphonium (10 passes), the kNN of PROTOCOL_CT (1 pass), the attempt-1 choice of
#   PROTOCOL_OK (3 passes: upright s1 with the L0/L1/L2 features, R180 s1, upright s7; nearest-bank cosine in L012, its 50 nearest bank
#   images, residual Mahalanobis of the upright s1/s7 errors, the R180 atom, empirical p-values and Fisher's sum) and the attempt-2 choice
#   (1 pass: nearest cosine in L012 and the mean of the 1000 largest cosines in L1, p-values, Fisher).  Same model, batch and precision as
#   PROTOCOL_CT's bench, 2000 ID test images, two warm-up batches, best of three timed passes; bank features (and bank errors) are
#   computed beforehand and kept on the GPU (not timed); calibration arrays are random placeholders of the right size (the cost of the
#   look-ups does not depend on their values).   usage: [DG_FP16=0] python code/ok_bench.py --tag edmc10 --gpu 0
import os, sys, time, json, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgcore import *
from ct_extract_defs import SPEC, SIGS, groups_for, bank_indices

p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--gpu', type=int, default=0)
p.add_argument('--n', type=int, default=2000); p.add_argument('--reps', type=int, default=3)
a = p.parse_args()
spec, bankset, bs = SPEC[a.tag]; bs = {500: 250, 250: 250, 125: 125}[bs]          # PROTOCOL_CT bench batches: 250 (32 px) / 125 (64 px)
dev = torch.device(f'cuda:{a.gpu}'); torch.backends.cudnn.benchmark = True
den, net = load_model(spec, dev)
G = groups_for(net, den, bankset, dev); H = [G['L0'][0], G['L1'][0], G['L2'][0]]
mods = dict(net.named_modules()); store = {}; active = [False]


def grab(i):
    def f(m, inp, o):
        if not active[0]: return
        if isinstance(o, (tuple, list)): o = o[0]
        o = o.float(); store[i] = torch.nn.functional.normalize(torch.cat([o.mean((2, 3)), o.std((2, 3), correction=0)], 1), dim=1)
    return f


for i, n in enumerate(H): mods[n].register_forward_hook(grab(i))


@torch.no_grad()
def passv(x, k, t='id', feats=False):
    sig = SIGS[k]; eps = shared_noise(x.shape[1:], 0, k, 0, dev); xn = x + sig * eps; f, fi = TF[t]
    store.clear(); active[0] = feats; d = fi(den(f(xn), sig)); active[0] = False
    e = (eps - (xn - d) / sig).square().flatten(1).sum(1)
    return e, ([store[i] for i in range(3)] if feats else None)


full = np.load(f'{NPY}/{bankset}.npy', mmap_mode='r'); bidx, _ = bank_indices(len(full)); xb = full[bidx]
BF = [[], [], []]; BE0, BE6 = [], []
with torch.no_grad():
    for i in range(0, len(xb), bs):
        x = to_t(np.ascontiguousarray(xb[i:i + bs]), dev)
        e0, F = passv(x, 0, feats=True); e6, _ = passv(x, 6)
        for l in range(3): BF[l].append(F[l])
        BE0.append(e0); BE6.append(e6)
BF = [torch.cat(b).T.contiguous() for b in BF]; BU = torch.stack([torch.log(torch.cat(BE0)), torch.log(torch.cat(BE6))], 1)
nb = BF[0].shape[1]; R1000 = min(1000, nb)
g = torch.Generator(device='cpu').manual_seed(0)
CAL = {k: torch.sort(torch.randn(2000, generator=g)).values.to(dev) for k in ('c', 'a', 'm', 'c2', 'p2')}
MU = torch.zeros(2, device=dev); PREC = torch.eye(2, device=dev)


def lp_low(cal, x): return torch.log((1 + torch.searchsorted(cal, x.contiguous(), right=True).float()) / (len(cal) + 1))
def lp_high(cal, x): return torch.log((1 + len(cal) - torch.searchsorted(cal, x.contiguous(), right=False).float()) / (len(cal) + 1))


@torch.no_grad()
def knn(x):
    _, F = passv(x, 0, feats=True)
    return (F[1] @ BF[1]).topk(10, dim=1).values[:, 9]


@torch.no_grad()
def att1(x):
    e0, F = passv(x, 0, feats=True); e180, _ = passv(x, 0, 'rot180'); e6, _ = passv(x, 6)
    S = (F[0] @ BF[0] + F[1] @ BF[1] + F[2] @ BF[2]) / 3; t = S.topk(50, dim=1)
    c = t.values[:, 0]; nn = t.indices
    U = torch.stack([torch.log(e0), torch.log(e6)], 1); Rr = U - BU[nn].mean(1); Z = Rr - MU
    m = torch.einsum('ij,jk,ik->i', Z, PREC, Z); at = torch.log(e180) - torch.log(e0)
    return lp_low(CAL['c'], c) + lp_low(CAL['a'], at) + lp_high(CAL['m'], m)


@torch.no_grad()
def att2(x):
    _, F = passv(x, 0, feats=True)
    c = ((F[0] @ BF[0] + F[1] @ BF[1] + F[2] @ BF[2]) / 3).max(1).values
    p = (F[1] @ BF[1]).topk(R1000, dim=1).values.mean(1)
    return lp_low(CAL['c2'], c) + lp_low(CAL['p2'], p)


@torch.no_grad()
def euphonium(x):
    s, up = 0, {}
    for T, k in (('rot90', 0), ('rot90', 6), ('rot180', 0)):
        sig = SIGS[k]; f, fi = TF[T]; eT = 0; eI = 0
        for j in (0, 1):
            eps = shared_noise(x.shape[1:], 0, k, j, dev); xn = x + sig * eps
            if (k, j) not in up: up[(k, j)] = (eps - (xn - den(xn, sig)) / sig).square().flatten(1).sum(1)
            eT = eT + (eps - (xn - fi(den(f(xn), sig))) / sig).square().flatten(1).sum(1); eI = eI + up[(k, j)]
        s = s + torch.log(eT / eI)
    return s


IDS = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'edmc100': 'cifar100_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
xt = np.load(f'{NPY}/{IDS[a.tag]}.npy', mmap_mode='r')[:a.n]
xs = [to_t(np.ascontiguousarray(xt[i:i + bs]), dev) for i in range(0, len(xt), bs)]


def bench(fn):
    for x in xs[:2]: fn(x)
    torch.cuda.synchronize(); best = 0.0
    for r in range(a.reps):
        t0 = time.time()
        for x in xs: fn(x)
        torch.cuda.synchronize(); best = max(best, len(xt) / (time.time() - t0))
    return best


res = dict(tag=a.tag, gpu=torch.cuda.get_device_name(a.gpu), fp16=FP16, batch=bs, n=len(xt), bank=nb)
for name, fn in (('euphonium', euphonium), ('knn', knn), ('att1', att1), ('att2', att2)): res[name + '_img_s'] = round(bench(fn), 1)
res['ms_per_img'] = {k[:-6]: round(1000 / v, 3) for k, v in res.items() if k.endswith('_img_s')}
print(json.dumps(res), flush=True)
os.makedirs(f'{ROOT}/runs/ok/bench', exist_ok=True); json.dump(res, open(f'{ROOT}/runs/ok/bench/{a.tag}.json', 'w'), indent=1)
