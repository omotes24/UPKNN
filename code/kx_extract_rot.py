# kx_extract_rot.py -- pooled decoder features (as ok_extract.py: L0, L1, L2; [g mean C, g std C, 2x2 grid means 4C, grid stds 4C])
#   of the ROTATED views of the bank with the noise NOT rotated: V_r = R_r(x) + s_k eps_k (eps_k = the rails' shared noise image of the
#   seed, the same un-rotated pattern for every view), so that the noise pattern cannot reveal the rotation.  View 0 equals ok_extract's.
#   output: runs/kx/rotb/s<seed>/<tag>/k<k>/bank__L<l>.npy [N, 4, 10C] float16 (+ idx)
#   usage: python code/kx_extract_rot.py --tag edmc10 --gpu 0 --ks 0,6 [--seed 0]
import os, sys, json, time, argparse
import numpy as np
import torch
from numpy.lib.format import open_memmap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgcore import ROOT, NPY, load_model, to_t, TF, shared_noise
from ct_extract_defs import SPEC, SIGS, groups_for, bank_indices
p = argparse.ArgumentParser()
p.add_argument('--tag', required=True); p.add_argument('--gpu', type=int, default=0); p.add_argument('--ks', default='0,6'); p.add_argument('--seed', type=int, default=0); p.add_argument('--hflip', type=int, default=0); p.add_argument('--src', default=''); p.add_argument('--n', type=int, default=0); p.add_argument('--name', default='bank')
a = p.parse_args()
spec, bankset, bs = SPEC[a.tag]
dev = torch.device(f'cuda:{a.gpu}'); torch.backends.cudnn.benchmark = True
den, net = load_model(spec, dev)
G = groups_for(net, den, bankset, dev)
HOOKS = [G['L0'][0], G['L1'][0], G['L2'][0]]
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
t0 = time.time()
if a.src:
    full = np.load(f'{NPY}/{a.src}.npy', mmap_mode='r'); bidx = np.sort(np.random.RandomState(7).choice(len(full), a.n, replace=False)); xu = full[bidx]
else:
    full = np.load(f'{NPY}/{bankset}.npy', mmap_mode='r'); bidx, _ = bank_indices(len(full)); xu = full[bidx]
if a.hflip: xu = np.ascontiguousarray(xu[:, :, ::-1])            # uint8 [N, H, W, C]: mirror the width axis
N = len(xu)
for k in [int(s) for s in a.ks.split(',')]:
    sig = SIGS[k]; od = f'{ROOT}/runs/kx/rotb/s{a.seed}/{a.tag}/k{k}'; os.makedirs(od, exist_ok=True)
    if os.path.exists(f'{od}/{a.name}__idx.npy'): print('exists', od, flush=True); continue
    M = None
    with torch.no_grad():
        for i in range(0, N, bs):
            x = to_t(np.ascontiguousarray(xu[i:i + bs]), dev)
            eps = shared_noise(x.shape[1:], a.seed, k, 0, dev)
            for r, t in enumerate(ROT):
                f, finv = TF[t]; store.clear(); den(f(x) + sig * eps, sig)
                P = [pool(store[l]) for l in range(3)]
                if M is None: M = {l: open_memmap(f'{od}/{a.name}__L{l}.npy.part', 'w+', np.float16, (N, 4, P[l].shape[1])) for l in range(3)}
                for l in range(3): M[l][i:i + len(x), r] = P[l].cpu().numpy()
    for l in range(3): M[l].flush(); del M[l]; os.replace(f'{od}/{a.name}__L{l}.npy.part', f'{od}/{a.name}__L{l}.npy')
    np.save(f'{od}/{a.name}__idx.npy', np.asarray(bidx, np.int64)); print(a.tag, 'k', k, f'{time.time() - t0:.0f}s', flush=True)
print('KX_EXTRACT_ROT_DONE', a.tag, f'{time.time() - t0:.0f}s', flush=True)
