# vl_fextract.py -- Vertical Limit: global [channel mean, channel std] of ONE decoder level (the last decoder block at resolution
#   R >> level, as ct_extract_defs.groups_for) at the noise level sigma_k (16-point grid) for the bank (upright view only) and the image
#   sets of runs/ok/sets_<tag>.json with the given prefixes.  Noisy input x + sigma_k eps_k with the shared noise image of the seed
#   (dgcore.shared_noise(..., seed, k, 0)), exactly as ok_extract.py / vl_extract.py.  final2 sets are processed only when
#   runs/vl/PROTOCOL_VL.json exists and its md5 equals --frozen (the protocol is fixed before the single final evaluation).
#   output: runs/vl/featf/s<seed>/<tag>/k<k>/<name>__L<level>.npy  [N, 2C] float16, <name>__idx.npy
#   usage: python code/vl_fextract.py --tag in64 --gpu 0 --seed 0 --k 0 --level 3 --prefix fresh-,devall-,sel- [--frozen <md5>]
import os, sys, json, time, re, hashlib, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgcore import ROOT, NPY, load_model, to_t, TF, shared_noise
from ct_extract_defs import SPEC, SIGS, PATS, part, bank_indices

p = argparse.ArgumentParser()
p.add_argument('--tag', required=True); p.add_argument('--gpu', type=int, default=0); p.add_argument('--seed', type=int, default=0)
p.add_argument('--k', type=int, required=True); p.add_argument('--level', type=int, required=True)
p.add_argument('--prefix', default='fresh-,devall-,sel-'); p.add_argument('--frozen', default=''); p.add_argument('--nobank', type=int, default=0)
a = p.parse_args()
if 'final2' in a.prefix:
    pf = f'{ROOT}/runs/vl/PROTOCOL_VL.json'
    assert os.path.exists(pf) and hashlib.md5(open(pf, 'rb').read()).hexdigest() == a.frozen, 'final2 needs the frozen protocol (md5)'
spec, bankset, bs = SPEC[a.tag]
dev = torch.device(f'cuda:{a.gpu}'); torch.backends.cudnn.benchmark = True
den, net = load_model(spec, dev)
mods = dict(net.named_modules())


@torch.no_grad()
def level_hook():
    names = [n for n in mods if any(re.search(q, n) for q in PATS)]; order, res, hs = [], {}, []
    def grab(name):
        def f(m, i, o):
            if isinstance(o, (tuple, list)): o = o[0]
            if torch.is_tensor(o) and o.ndim == 4:
                if name not in order: order.append(name)
                res[name] = int(o.shape[-1])
        return f
    for n in names: hs.append(mods[n].register_forward_hook(grab(n)))
    x = to_t(np.ascontiguousarray(np.load(f'{NPY}/{bankset}.npy', mmap_mode='r')[:2]), dev); den(x, 0.2); Rz = int(x.shape[-1])
    for h in hs: h.remove()
    dec = [n for n in order if n in res and part(n) == 'dec' and res[n] == (Rz >> a.level)]
    return dec[-1]


HOOK = level_hook(); store = {}
def grab(m, i, o):
    if isinstance(o, (tuple, list)): o = o[0]
    store['o'] = o
mods[HOOK].register_forward_hook(grab)
print(a.tag, 'level', a.level, 'hook', HOOK, 'sigma', SIGS[a.k], flush=True)


@torch.no_grad()
def run(xu, idx, pre, od, name):
    if os.path.exists(f'{od}/{name}__idx.npy'): print('exists', name, flush=True); return
    out = []; sig = SIGS[a.k]
    for i in range(0, len(xu), bs):
        x = to_t(np.ascontiguousarray(xu[i:i + bs]), dev)
        if pre != 'none': x = TF[pre][0](x)
        eps = shared_noise(x.shape[1:], a.seed, a.k, 0, dev); store.clear(); den(x + sig * eps, sig)
        o = store['o'].float(); out.append(torch.cat([o.mean((2, 3)), o.std((2, 3), correction=0)], 1).half().cpu().numpy())
    np.save(f'{od}/{name}__L{a.level}.npy', np.concatenate(out)); np.save(f'{od}/{name}__idx.npy', np.asarray(idx, np.int64))


od = f'{ROOT}/runs/vl/featf/s{a.seed}/{a.tag}/k{a.k}'; os.makedirs(od, exist_ok=True); t0 = time.time()
if not a.nobank:
    full = np.load(f'{NPY}/{bankset}.npy', mmap_mode='r'); bidx, _ = bank_indices(len(full))
    run(full[bidx], bidx, 'none', od, 'bank'); print(f'{a.tag} s{a.seed} k{a.k} bank {len(bidx)} {time.time() - t0:.0f}s', flush=True)
for name, data, idx, pre in [s for s in json.load(open(f'{ROOT}/runs/ok/sets_{a.tag}.json')) if s[0].startswith(tuple(a.prefix.split(',')))]:
    run(np.load(f'{NPY}/{data}.npy', mmap_mode='r')[np.asarray(idx, dtype=np.int64)], idx, pre, od, name)
print('VL_FEXTRACT_DONE', a.tag, a.seed, a.k, a.level, f'{time.time() - t0:.0f}s', flush=True)
