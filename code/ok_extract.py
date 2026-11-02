# ok_extract.py -- pooled decoder features of rotated views (PROTOCOL_OK.md, the oriented-kNN family)
#   For every image x (uint8 -> [-1, 1]; --pre applied first) of the bank (the ID training images of ct_extract.py's 50k bank) and of
#   the image sets of a sets json ([[name, data, idx, pre], ...]), the four views V_r = R_r(x + s_k eps_k), r = 0..3 (rotation by
#   0/90/180/270 degrees as in dgcore.TF; eps_k = the rails' shared noise image of seed --seed and scale index k, +eps only; s_k the
#   16-point grid value used by ct_extract.py) are passed through the denoiser once each.  Stored per view:
#     the outputs of the last decoder blocks at full, half and quarter resolution (hooks L0, L1, L2 of ct_extract.py) pooled as
#       [global channel mean (C), global channel std (C), 2x2-grid channel means (4C, quadrants TL, TR, BL, BR), grid stds (4C)]  float16
#     the squared noise-prediction error in the upright frame  e = ||eps - R_r^-1 epshat(V_r)||^2  (float64; Euphonium's e_k with +eps)
#   output: runs/ok/s<seed>/<tag>/k<k>/<name>__L<l>.npy [N, 4, 10C] float16, <name>__e.npy [N, 4], <name>__idx.npy [N]
#   usage: [DG_FP16=0] python code/ok_extract.py --tag edmc10 --gpu 0 --sets runs/ok/sets_edmc10.json [--ks 0,6] [--seed 0] [--only a,b]
import os, sys, json, time, argparse
import numpy as np
import torch
from numpy.lib.format import open_memmap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgcore import ROOT, NPY, load_model, to_t, TF, shared_noise
from ct_extract_defs import SPEC, SIGS, groups_for, bank_indices

p = argparse.ArgumentParser()
p.add_argument('--tag', required=True)
p.add_argument('--gpu', type=int, default=0)
p.add_argument('--sets', required=True)
p.add_argument('--ks', default='0,6')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--only', default='', help='comma-separated set names (bank included as "bank"); default all + bank')
p.add_argument('--out', default='runs/ok')
a = p.parse_args()
spec, bankset, bs = SPEC[a.tag]
dev = torch.device(f'cuda:{a.gpu}')
torch.backends.cudnn.benchmark = True
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
    qm = q.mean((3, 5)).permute(0, 2, 3, 1).reshape(B, 4 * C)
    qs = q.std((3, 5), correction=0).permute(0, 2, 3, 1).reshape(B, 4 * C)
    return torch.cat([o.mean((2, 3)), o.std((2, 3), correction=0), qm, qs], 1).half()


@torch.no_grad()
def run(xu, idx, pre, k, od, name):
    sig = SIGS[k]; N = len(xu); fn = {l: f'{od}/{name}__L{l}.npy' for l in range(3)}
    if os.path.exists(f'{od}/{name}__idx.npy'): print('exists', od, name, flush=True); return
    M, E = None, np.zeros((N, 4), np.float64)
    for i in range(0, N, bs):
        x = to_t(np.ascontiguousarray(xu[i:i + bs]), dev)
        if pre != 'none': x = TF[pre][0](x)
        eps = shared_noise(x.shape[1:], a.seed, k, 0, dev); xn = x + sig * eps
        for r, t in enumerate(ROT):
            f, finv = TF[t]; store.clear()
            d = finv(den(f(xn), sig)); eh = (xn - d) / sig
            E[i:i + len(x), r] = (eps - eh).double().square().flatten(1).sum(1).cpu().numpy()
            P = [pool(store[l]) for l in range(3)]
            if M is None: M = {l: open_memmap(fn[l] + '.part', 'w+', np.float16, (N, 4, P[l].shape[1])) for l in range(3)}
            for l in range(3): M[l][i:i + len(x), r] = P[l].cpu().numpy()
    for l in range(3): M[l].flush(); del M[l]; os.replace(fn[l] + '.part', fn[l])
    np.save(f'{od}/{name}__e.npy', E); np.save(f'{od}/{name}__idx.npy', np.asarray(idx, np.int64))


SETS = json.load(open(f'{ROOT}/{a.sets}'))
only = set(a.only.split(',')) if a.only else None
meta = dict(tag=a.tag, spec=spec, hooks=HOOKS, seed=a.seed, fp16=os.environ.get('DG_FP16', '1'), bank=bankset, times={})
t0 = time.time()
for k in [int(s) for s in a.ks.split(',')]:
    od = f'{ROOT}/{a.out}/s{a.seed}/{a.tag}/k{k}'; os.makedirs(od, exist_ok=True)
    if only is None or 'bank' in only:
        full = np.load(f'{NPY}/{bankset}.npy', mmap_mode='r'); bidx, _ = bank_indices(len(full))
        t1 = time.time(); run(full[bidx], bidx, 'none', k, od, 'bank'); meta['times'][f'bank_k{k}'] = round(time.time() - t1, 1)
        print(f'{a.tag} k{k} bank {len(bidx)} {time.time() - t1:.0f}s', flush=True)
    for name, data, idx, pre in SETS:
        if only is not None and name not in only: continue
        t1 = time.time()
        x = np.load(f'{NPY}/{data}.npy', mmap_mode='r')[np.asarray(idx, dtype=np.int64)]
        run(x, idx, pre, k, od, name); meta['times'][f'{name}_k{k}'] = round(time.time() - t1, 1)
        print(f'{a.tag} k{k} {name} {len(idx)} {time.time() - t1:.0f}s', flush=True)
    json.dump(meta, open(f'{od}/meta_{os.getpid()}.json', 'w'), indent=1)
print('OK_EXTRACT_DONE', a.tag, a.seed, f'{time.time() - t0:.0f}s', flush=True)
