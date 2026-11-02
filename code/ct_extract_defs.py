# ct_extract_defs.py -- the definitions of ct_extract.py (models, hooks, groups, bank images) as functions, for ct_bench.py
import re
import numpy as np
import torch
from dgcore import NPY, to_t

SPEC = {
    'edmc10':   ('edm:edm-cifar10-32x32-uncond-vp.pkl', 'cifar10_train', 500),
    'l6svhn':   ('l6:layer6_svhn_vpsde.pt', 'svhn_train', 500),
    'dpceleba': ('iddpm:diffpath_celeba32.pt', 'celeba_train', 500),
    'edmc100':  ('edmsd:edm_cifar100_uncond_vp.pt', 'cifar100_train', 250),
    'in64':     ('edm2:edm2-img64-xs-uncond-2147483-0.100.pkl', 'in64_imagenet_train10', 125),
    'fmnist':   ('own:models/fmnist_s/final.pt', 'g32_fmnist_train', 500),
}
PATS = [r'(^|\.)(enc|dec)\.\d+x\d+_(conv|block\d+|in\d+|up|down)$',
        r'^(down_blocks\.\d+|mid_block|up_blocks\.\d+)$',
        r'^(input_blocks\.\d+|middle_block|output_blocks\.\d+)$']
S16 = np.exp(np.linspace(np.log(0.01), np.log(31.6228), 16))
SIGS = [float('%.6g' % s) for s in S16]
GROUPS = ['L0', 'L1', 'L2', 'L0+L1', 'L0+L2', 'L1+L2', 'L0+L1+L2', 'DEC', 'ENC', 'ALL']


def part(n):
    if re.search(r'(^|\.)enc\.', n) or n.startswith('down_blocks') or n.startswith('input_blocks'): return 'enc'
    if re.search(r'(^|\.)dec\.', n) or n.startswith('up_blocks') or n.startswith('output_blocks'): return 'dec'
    return 'mid'


@torch.no_grad()
def groups_for(net, den, bankset, dev):
    """the hook groups of ct_extract.py (dry run with temporary hooks, forward order)"""
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
    def last_dec(r): return [n for n in names if part(n) == 'dec' and res[n] == r][-1]
    L = [last_dec(R >> l) for l in range(3)]
    return {'L0': [L[0]], 'L1': [L[1]], 'L2': [L[2]], 'L0+L1': [L[0], L[1]], 'L0+L2': [L[0], L[2]], 'L1+L2': [L[1], L[2]],
            'L0+L1+L2': L, 'DEC': [n for n in names if part(n) == 'dec'], 'ENC': [n for n in names if part(n) == 'enc'], 'ALL': list(names)}


def bank_indices(Nf, nb_max=50000, nb_small=10000):
    """(indices of the 50k bank in the training set, positions of the nested 10k bank within it) as in ct_extract.py"""
    nb = min(nb_max, Nf)
    bidx = np.sort(np.random.RandomState(0).choice(Nf, nb, replace=False)) if nb < Nf else np.arange(Nf)
    small = np.sort(np.random.RandomState(1).permutation(nb)[:min(nb_small, nb)])
    return bidx, small
