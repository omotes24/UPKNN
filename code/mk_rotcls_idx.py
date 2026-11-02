# mk_rotcls_idx.py -- the index file of rotcls_eval.py for the ood1k- sets: {"in64": {"__id__": idx, "<data>": idx, ...}}
import os, json
ROOT = os.path.expanduser('~/upknn')
S = [r for r in json.load(open(f'{ROOT}/runs/ok/sets_in64.json')) if r[0].startswith('ood1k-')]
com = {}
for name, data, idx, pre in S:
    com['__id__' if data == 'in64_imagenet_test' else data] = idx
os.makedirs(f'{ROOT}/runs/ood1k', exist_ok=True); json.dump({'in64': com}, open(f'{ROOT}/runs/ood1k/rotcls_idx.json', 'w'))
print({k: len(v) for k, v in com.items()})
