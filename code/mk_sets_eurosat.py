# mk_sets_eurosat.py -- runs/ok/sets_eurosat.json (PROTOCOL_E3.md): [name, data, idx, pre]; prefix e3-
import os, json
import numpy as np
ROOT = os.path.expanduser('~/upknn'); OUT = f'{ROOT}/runs/ok/sets_eurosat.json'
assert not os.path.exists(OUT), 'exists'
n = len(np.load(f'{ROOT}/data/npy/eurosat_test.npy', mmap_mode='r'))
idx = sorted(int(i) for i in np.random.RandomState(0).choice(n, 1000, replace=False))
S = [['e3-eurosat_test', 'eurosat_test', idx, 'none'], ['e3-eurosat_test__vflip', 'eurosat_test', idx, 'vflip'], ['e3-eurosat_test__rot90', 'eurosat_test', idx, 'rot90']]
C10 = {r[1]: r[2] for r in json.load(open(f'{ROOT}/runs/ok/sets_edmc10.json')) if r[0].startswith('final2-') and r[3] == 'none'}
for s in ['cifar10_test', 'cifar100_test', 'svhn_test', 'celeba_test', 'textures', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val']:
    S.append([f'e3-{s}', s, [int(i) for i in C10[s]], 'none'])
json.dump(S, open(OUT, 'w')); print([(r[0], len(r[2]), r[3]) for r in S])
