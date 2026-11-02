# mk_sets_e4.py -- runs/ok/sets_<tag>.json for the tiers of PROTOCOL_E4.md ([name, data, idx, pre], prefix e4-) and the index files of
#   the comparison runs (eu/runs/jobs/e4_idx/<tag>__<set>.json; '<id>__cal' = 1,000 ID training images outside the bank)
import os, json
import numpy as np
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ct_extract_defs import bank_indices
ROOT = os.path.expanduser('~/upknn'); NPY = f'{ROOT}/data/npy'; IDX = f'{ROOT}/eu/runs/jobs/e4_idx'
os.makedirs(IDX, exist_ok=True)
def n_of(name): return len(np.load(f'{NPY}/{name}.npy', mmap_mode='r'))
def pick(name, n): N = n_of(name); return sorted(int(i) for i in np.random.RandomState(0).choice(N, min(n, N), replace=False))
TF = ('vflip', 'rot90', 'rot180', 'hflip')
C10 = {r[1]: [int(i) for i in r[2]] for r in json.load(open(f'{ROOT}/runs/ok/sets_edmc10.json')) if r[0].startswith('final2-') and r[3] == 'none'}
NAT10 = ['cifar10_test', 'cifar100_test', 'svhn_test', 'celeba_test', 'textures', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val']
TIERS = {}
# A. chest X-rays (grey)
i = pick('g32_chest_test', 1000)
S = [['e4-g32_chest_test', 'g32_chest_test', i, 'none']] + [[f'e4-g32_chest_test__{t}', 'g32_chest_test', i, t] for t in TF]
for s in ['g32_organa_test', 'g32_path_gray_test', 'g32_pneumonia_all', 'g32_cifar10_gray', 'g32_svhn_gray', 'g32_mnist_test', 'g32_fmnist_test']:
    S.append([f'e4-{s}', s, pick(s, 500), 'none'])
TIERS['chest'] = (S, 'g32_chest_train')
# B. histology (RGB)
i = pick('path_test', 1000)
S = [['e4-path_test', 'path_test', i, 'none']] + [[f'e4-path_test__{t}', 'path_test', i, t] for t in TF]
for s in ['blood_all', 'derma_all', 'chest_rgb_test']: S.append([f'e4-{s}', s, pick(s, 500), 'none'])
for s in NAT10: S.append([f'e4-{s}', s, C10[s], 'none'])
TIERS['path'] = (S, 'path_train')
# C. CIFAR-10 generators trained with / without rotations: the final2 images of the CIFAR-10 tier
S = [['e4-cifar10_test', 'cifar10_test', C10['cifar10_test'], 'none'], ['e4-cifar10_test__vflip', 'cifar10_test', C10['cifar10_test'], 'vflip']]
for s in NAT10[1:]: S.append([f'e4-{s}', s, C10[s], 'none'])
for t in ('c10n4', 'c10x4', 'c10r25', 'c10r50', 'c10r100'): TIERS[t] = (S, 'cifar10_train')
for tag, (S, train) in TIERS.items():
    out = f'{ROOT}/runs/ok/sets_{tag}.json'
    if os.path.exists(out): assert json.load(open(out)) == S, f'{out} exists and differs'
    else: json.dump(S, open(out, 'w'))
    for name, data, idx, pre in S: json.dump(idx, open(f'{IDX}/{tag}__{name[3:]}.json', 'w'))
    bidx, _ = bank_indices(n_of(train)); rest = np.setdiff1d(np.arange(n_of(train)), bidx)
    cal = sorted(int(j) for j in np.random.RandomState(1).choice(rest, 1000, replace=False)) if len(rest) >= 1000 else []
    if cal: json.dump(cal, open(f'{IDX}/{tag}__{train}__cal.json', 'w'))
    print(tag, len(S), 'sets', [(r[0], len(r[2]), r[3]) for r in S][:3], '... cal', len(cal), flush=True)
