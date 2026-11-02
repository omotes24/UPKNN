# ct_extract.py -- content statistics of PROTOCOL_CT.md: pooled (channel mean, std) activations of every U-Net block on the
#   upright noisy input x +/- s_k eps_k (the rails' shared noise, seed 0, scale index k), nearest-neighbour similarity to a bank
#   of ID training images for hook groups (mean over the group's hooks of the cosine between L2-normalised pooled features).
#   output: runs/ct/<tag>/k<k>/<set>.npz   K [N, G, 2 variants (plus, pair), 2 banks (10k, 50k), 2 kk (1, 10)] float32, idx
#           runs/ct/<tag>/meta.json        groups, hooks, bank, timings
#   usage: DG_FP16=... python code/ct_extract.py --tag edmc10 --gpu 0 [--ks 0,1,...,9]
import os, sys, re, time, json, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgcore import *

SPEC = {   # tag: model spec, bank set (ID training images), batch size
    'edmc10':   ('edm:edm-cifar10-32x32-uncond-vp.pkl', 'cifar10_train', 500),
    'l6svhn':   ('l6:layer6_svhn_vpsde.pt', 'svhn_train', 500),
    'dpceleba': ('iddpm:diffpath_celeba32.pt', 'celeba_train', 500),
    'edmc100':  ('edmsd:edm_cifar100_uncond_vp.pt', 'cifar100_train', 250),
    'in64':     ('edm2:edm2-img64-xs-uncond-2147483-0.100.pkl', 'in64_imagenet_train10', 125),
    'fmnist':   ('own:models/fmnist_s/final.pt', 'g32_fmnist_train', 500),
}
PATS = [r'(^|\.)(enc|dec)\.\d+x\d+_(conv|block\d+|in\d+|up|down)$',      # SongUNet (EDM, own models) and the EDM2 U-Net
        r'^(down_blocks\.\d+|mid_block|up_blocks\.\d+)$',                # diffusers UNet2DModel (layer6 SVHN)
        r'^(input_blocks\.\d+|middle_block|output_blocks\.\d+)$']        # improved-diffusion U-Net (DiffPath CelebA)
S16 = np.exp(np.linspace(np.log(0.01), np.log(31.6228), 16))
SIGS = [float('%.6g' % s) for s in S16]                                  # the exact values of the rails runs
GROUPS = ['L0', 'L1', 'L2', 'L0+L1', 'L0+L2', 'L1+L2', 'L0+L1+L2', 'DEC', 'ENC', 'ALL']
NB_MAX, NB_SMALL = 50000, 10000

p = argparse.ArgumentParser()
p.add_argument('--tag', required=True)
p.add_argument('--gpu', type=int, default=0)
p.add_argument('--ks', default='0,1,2,3,4,5,6,7,8,9')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--qchunk', type=int, default=500)
p.add_argument('--out', default='runs/ct')
a = p.parse_args()
spec, bankset, bs = SPEC[a.tag]
dev = torch.device(f'cuda:{a.gpu}')
torch.backends.cudnn.benchmark = True
od = f'{ROOT}/{a.out}/{a.tag}'; os.makedirs(od, exist_ok=True)
den, net = load_model(spec, dev)
mods = dict(net.named_modules())
names = [n for n in mods if any(re.search(q, n) for q in PATS)]
store, order = {}, []


def grab(name):
    def f(m, i, o):
        if isinstance(o, (tuple, list)): o = o[0]
        if torch.is_tensor(o) and o.ndim == 4:
            if name not in order: order.append(name)
            o = o.float(); store[name] = (torch.cat([o.mean((2, 3)), o.std((2, 3), correction=0)], 1), o.shape[-1])
    return f


for n in names: mods[n].register_forward_hook(grab(n))
full = np.load(f'{NPY}/{bankset}.npy', mmap_mode='r')
with torch.no_grad(): den(to_t(np.ascontiguousarray(full[:2]), dev), 0.2)
R = int(to_t(np.ascontiguousarray(full[:1]), dev).shape[-1])
names = [n for n in order if n in store]                                 # forward order
res = {n: int(store[n][1]) for n in names}


def part(n):
    if re.search(r'(^|\.)enc\.', n) or n.startswith('down_blocks') or n.startswith('input_blocks'): return 'enc'
    if re.search(r'(^|\.)dec\.', n) or n.startswith('up_blocks') or n.startswith('output_blocks'): return 'dec'
    return 'mid'


def last_dec(r):
    c = [n for n in names if part(n) == 'dec' and res[n] == r]
    assert c, (a.tag, r); return c[-1]


L = [last_dec(R >> l) for l in range(3)]
G = {'L0': [L[0]], 'L1': [L[1]], 'L2': [L[2]], 'L0+L1': [L[0], L[1]], 'L0+L2': [L[0], L[2]], 'L1+L2': [L[1], L[2]],
     'L0+L1+L2': L, 'DEC': [n for n in names if part(n) == 'dec'], 'ENC': [n for n in names if part(n) == 'enc'], 'ALL': list(names)}
member = {n: [gi for gi, g in enumerate(GROUPS) if n in G[g]] for n in names}
use = [n for n in names if member[n]]
print(a.tag, 'R', R, 'hooks', len(names), {g: G[g] if len(G[g]) <= 3 else len(G[g]) for g in GROUPS}, flush=True)

# bank: min(50k, all) ID training images (seed 0); the 10k bank = first 10k of a fixed permutation (seed 1) of it (nested)
Nf = len(full); nb = min(NB_MAX, Nf)
bidx = np.sort(np.random.RandomState(0).choice(Nf, nb, replace=False)) if nb < Nf else np.arange(Nf)
small = np.sort(np.random.RandomState(1).permutation(nb)[:min(NB_SMALL, nb)])
xb = np.ascontiguousarray(full[bidx]); small_t = torch.from_numpy(small).to(dev)
SETS = json.load(open(f'{ROOT}/{a.out}/sets_{a.tag}.json'))
meta = dict(tag=a.tag, spec=spec, R=R, hooks={n: dict(res=res[n], part=part(n), C=int(store[n][0].shape[1] // 2)) for n in names},
            groups={g: G[g] for g in GROUPS}, bank=dict(set=bankset, n_all=int(Nf), n=int(nb), n_small=int(len(small))),
            axes='K[N, group, variant(plus,pair), bank(10k,50k), kk(1,10)]', sigmas=[SIGS[k] for k in range(10)], times={})
json.dump(meta, open(f'{od}/meta.json', 'w'), indent=1)


@torch.no_grad()
def pooled(xu, k, pre='none'):
    """hook -> (plus, pair): L2-normalised pooled features [N, 2C] (fp16) of x + s eps and of the mean over x +/- s eps"""
    sig = SIGS[k]; fp = {n: [] for n in use}; fm = {n: [] for n in use}
    for i in range(0, len(xu), bs):
        x = to_t(np.ascontiguousarray(xu[i:i + bs]), dev)
        if pre != 'none': x = TF[pre][0](x)
        for j, out in ((0, fp), (1, fm)):
            eps = shared_noise(x.shape[1:], a.seed, k, j, dev)
            store.clear(); den(x + sig * eps, sig)
            for n in use: out[n].append(store[n][0].half())
    F = {}
    for n in use:
        P, M = torch.cat(fp.pop(n)).float(), torch.cat(fm.pop(n)).float()
        F[n] = (torch.nn.functional.normalize(P, dim=1).half(), torch.nn.functional.normalize((P + M) / 2, dim=1).half())
        del P, M
    return F


@torch.no_grad()
def knn(Q, B):
    """[N, G, 2 variants, 2 banks, 2 kk]: kk-th largest group similarity over the bank"""
    nq = Q[use[0]][0].shape[0]; out = np.zeros((nq, len(GROUPS), 2, 2, 2), np.float32)
    for c0 in range(0, nq, a.qchunk):
        c1 = min(nq, c0 + a.qchunk)
        for v in (0, 1):
            acc = [torch.zeros(c1 - c0, nb, device=dev) for _ in GROUPS]
            for n in use:
                S = Q[n][v][c0:c1].float() @ B[n][v].float().T
                for gi in member[n]: acc[gi] += S
                del S
            for gi, g in enumerate(GROUPS):
                A = acc[gi] / len(G[g])
                for bi, M in enumerate((A.index_select(1, small_t), A)):
                    top = M.topk(10, dim=1).values
                    out[c0:c1, gi, v, bi, 0] = top[:, 0].cpu().numpy(); out[c0:c1, gi, v, bi, 1] = top[:, 9].cpu().numpy()
            del acc
    return out


t0 = time.time()
for k in [int(s) for s in a.ks.split(',')]:
    kd = f'{od}/k{k}'
    if os.path.exists(f'{kd}/DONE'): print('done', k); continue
    os.makedirs(kd, exist_ok=True); t1 = time.time()
    B = pooled(xb, k); tb = time.time() - t1
    for name, data, idx, pre in SETS:
        fn = f'{kd}/{name}.npz'
        if os.path.exists(fn): continue
        x = np.load(f'{NPY}/{data}.npy', mmap_mode='r')[np.asarray(idx, dtype=np.int64)]
        K = knn(pooled(x, k, pre), B)
        np.savez(fn, K=K, idx=np.asarray(idx, dtype=np.int64))
    del B; torch.cuda.empty_cache()
    meta['times'][str(k)] = dict(bank_s=round(tb, 1), total_s=round(time.time() - t1, 1))
    json.dump(meta, open(f'{od}/meta.json', 'w'), indent=1)
    open(f'{kd}/DONE', 'w').write('ok\n')
    print(f'{a.tag} k={k} bank {tb:.0f}s, total {time.time() - t1:.0f}s, elapsed {time.time() - t0:.0f}s', flush=True)
print('CT_EXTRACT_DONE', a.tag, f'{time.time() - t0:.0f}s', flush=True)
