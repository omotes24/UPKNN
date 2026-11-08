# vl_extract.py -- Vertical Limit exploration: pooled decoder features (as ok_extract.py: L0, L1, L2 = last decoder block at full, 1/2,
#   1/4 resolution; [g mean C, g std C, 2x2 grid means 4C, grid stds 4C]) at several noise levels sigma_k of the 16-point grid.
#   Bank (ID training images of ct_extract.py's bank): the four rotated views R_r(x) + s_k eps_k with the noise NOT rotated
#   (view 0 = the upright image), [N, 4, 10C].  Sets (devall-, sel-, fresh- only; never final2): the upright view x + s_k eps_k, [N, 10C].
#   eps_k = the shared noise image of the seed and scale index k (dgcore.shared_noise(..., seed, k, 0)).
#   output: runs/vl/feat/s<seed>/<tag>/k<k>/<name>__L<l>.npy float16 (+ <name>__idx.npy)
#   usage: python code/vl_extract.py --tag edmc10 --gpu 0 --ks 0,1,2,3,4,5 [--seed 0] [--levels 0,1,2[,3]] [--bigbank npyname --nbig n]
import os, sys, json, time, argparse
import numpy as np
import torch
from numpy.lib.format import open_memmap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgcore import ROOT, NPY, load_model, to_t, TF, shared_noise
from ct_extract_defs import SPEC, SIGS, groups_for, bank_indices, PATS, part
import re



@torch.no_grad()
def deeper_levels(net, den, bankset, dev):
    """the last decoder block at 1/8 resolution (L3), found as groups_for does for L0-L2"""
    mods = dict(net.named_modules()); names = [n for n in mods if any(re.search(q, n) for q in PATS)]
    order, res, hs = [], {}, []
    def grab(name):
        def f(m, i, o):
            if isinstance(o, (tuple, list)): o = o[0]
            if torch.is_tensor(o) and o.ndim == 4:
                if name not in order: order.append(name)
                res[name] = int(o.shape[-1])
        return f
    for n in names: hs.append(mods[n].register_forward_hook(grab(n)))
    full = np.load(f'{NPY}/{bankset}.npy', mmap_mode='r'); x = to_t(np.ascontiguousarray(full[:2]), dev)
    den(x, 0.2); R = int(x.shape[-1])
    for h in hs: h.remove()
    names = [n for n in order if n in res]
    dec = [n for n in names if part(n) == 'dec' and res[n] == (R >> 3)]
    assert dec, 'no decoder block at 1/8 resolution'
    return {'L3': [dec[-1]]}


p = argparse.ArgumentParser()
p.add_argument('--tag', required=True); p.add_argument('--gpu', type=int, default=0); p.add_argument('--ks', default='0,1,2,3,4,5')
p.add_argument('--seed', type=int, default=0); p.add_argument('--prefix', default='devall-,sel-,fresh-')
p.add_argument('--skipbank', type=int, default=0); p.add_argument('--bigbank', default='', help='npy name of a larger bank source (e.g. in64_imagenet_train_hf)')
p.add_argument('--nbig', type=int, default=0); p.add_argument('--bigviews', type=int, default=4); p.add_argument('--bankhf', type=int, default=0); p.add_argument('--hfviews', type=int, default=1); p.add_argument('--levels', default='0,1,2', help='decoder levels (0 = full resolution, 3 = 1/8)')
a = p.parse_args()
assert 'final2' not in a.prefix
assert a.bigviews in (1, 4)
spec, bankset, bs = SPEC[a.tag]
dev = torch.device(f'cuda:{a.gpu}'); torch.backends.cudnn.benchmark = True
den, net = load_model(spec, dev)
G = groups_for(net, den, bankset, dev)
LEV = [int(v) for v in a.levels.split(',')]
if max(LEV) > 2: G.update(deeper_levels(net, den, bankset, dev))
HOOKS = [G[f'L{l}'][0] for l in LEV]
mods = dict(net.named_modules()); store = {}


def grab(i):
    def f(m, inp, o):
        if isinstance(o, (tuple, list)): o = o[0]
        store[i] = o
    return f


for i, n in enumerate(HOOKS): mods[n].register_forward_hook(grab(i))
ROT = ['id', 'rot90', 'rot180', 'rot270']


def pool(o):
    o = o.float(); B, C, H, W = o.shape
    q = o.reshape(B, C, 2, H // 2, 2, W // 2)
    qm = q.mean((3, 5)).permute(0, 2, 3, 1).reshape(B, 4 * C); qs = q.std((3, 5), correction=0).permute(0, 2, 3, 1).reshape(B, 4 * C)
    return torch.cat([o.mean((2, 3)), o.std((2, 3), correction=0), qm, qs], 1).half()


@torch.no_grad()
def run(xu, idx, pre, k, od, name, views):
    if os.path.exists(f'{od}/{name}__idx.npy') and all(os.path.exists(f'{od}/{name}__L{l}.npy') for l in LEV): print('exists', od, name, flush=True); return
    sig = SIGS[k]; N = len(xu); M = None
    for i in range(0, N, bs):
        x = to_t(np.ascontiguousarray(xu[i:i + bs]), dev)
        if pre != 'none': x = TF[pre][0](x)
        eps = shared_noise(x.shape[1:], a.seed, k, 0, dev)
        for r in range(views):
            store.clear(); den(TF[ROT[r]][0](x) + sig * eps, sig)
            P = {l: pool(store[j]) for j, l in enumerate(LEV)}
            if M is None:
                M = {l: open_memmap(f'{od}/{name}__L{l}.npy.part', 'w+', np.float16, (N, views, P[l].shape[1]) if views > 1 else (N, P[l].shape[1])) for l in LEV}
            for l in LEV:
                if views > 1: M[l][i:i + len(x), r] = P[l].cpu().numpy()
                else: M[l][i:i + len(x)] = P[l].cpu().numpy()
    for l in LEV: M[l].flush(); del M[l]; os.replace(f'{od}/{name}__L{l}.npy.part', f'{od}/{name}__L{l}.npy')
    np.save(f'{od}/{name}__idx.npy', np.asarray(idx, np.int64))


SETS = [s for s in json.load(open(f'{ROOT}/runs/ok/sets_{a.tag}.json')) if s[0].startswith(tuple(a.prefix.split(',')))]
t0 = time.time()
for k in [int(s) for s in a.ks.split(',')]:
    od = f'{ROOT}/runs/vl/feat/s{a.seed}/{a.tag}/k{k}'; os.makedirs(od, exist_ok=True)
    if not a.skipbank:
        full = np.load(f'{NPY}/{bankset}.npy', mmap_mode='r'); bidx, _ = bank_indices(len(full))
        run(full[bidx], bidx, 'none', k, od, 'bank', 4); print(f'{a.tag} k{k} bank {len(bidx)} {time.time() - t0:.0f}s', flush=True)
    if a.bankhf:                                                      # the bank's horizontal mirror images (upright view only)
        full = np.load(f'{NPY}/{bankset}.npy', mmap_mode='r'); bidx, _ = bank_indices(len(full))
        xu = np.ascontiguousarray(full[bidx][:, :, ::-1])
        run(xu, bidx, 'none', k, od, 'bankhf' + ('' if a.hfviews == 1 else '4'), a.hfviews); print(f'{a.tag} k{k} bankhf {len(bidx)} {time.time() - t0:.0f}s', flush=True)
    if a.bigbank:
        full = np.load(f'{NPY}/{a.bigbank}.npy', mmap_mode='r'); n = a.nbig or len(full)
        bidx = np.sort(np.random.RandomState(0).choice(len(full), n, replace=False)) if n < len(full) else np.arange(len(full))
        run(full[bidx], bidx, 'none', k, od, f'bigbank{n}' + ('' if a.bigviews == 4 else f'v{a.bigviews}'), a.bigviews); print(f'{a.tag} k{k} bigbank {n} {time.time() - t0:.0f}s', flush=True)
    for name, data, idx, pre in SETS:
        x = np.load(f'{NPY}/{data}.npy', mmap_mode='r')[np.asarray(idx, dtype=np.int64)]
        run(x, idx, pre, k, od, name, 1)
    print(f'{a.tag} k{k} sets {len(SETS)} {time.time() - t0:.0f}s', flush=True)
print('VL_EXTRACT_DONE', a.tag, a.seed, f'{time.time() - t0:.0f}s', flush=True)
