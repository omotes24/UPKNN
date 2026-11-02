# mk_e5.py -- PROTOCOL_E5.md: nested random ImageNet-64 banks from the Hugging Face training shard and the set lists of the bank tiers
import os, json
import numpy as np
ROOT = os.path.expanduser('~/upknn'); NPY = f'{ROOT}/data/npy'
X = np.load(f'{NPY}/in64_imagenet_train_hf.npy', mmap_mode='r'); perm = np.random.RandomState(0).permutation(len(X))
S = [r for r in json.load(open(f'{ROOT}/runs/ok/sets_in64.json')) if r[0].startswith('ood1k-')]
for k in (10, 25, 50):
    out = f'{NPY}/in64_hfbank_{k}k.npy'
    if not os.path.exists(out):
        np.save(out + '.tmp.npy', np.ascontiguousarray(X[np.sort(perm[:k * 1000])])); os.replace(out + '.tmp.npy', out)
    p = f'{ROOT}/runs/ok/sets_in64h{k}.json'
    if not os.path.exists(p): json.dump(S, open(p, 'w'))
    print('bank', k, 'k', np.load(out, mmap_mode='r').shape, 'sets', len(S), flush=True)
