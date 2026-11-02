# mk_sets_in256.py -- runs/ok/sets_in256.json for the ImageNet-256 tier (PROTOCOL_IN256.md): [name, data, idx, pre]
#   h256-*: ID (final-split images of the stored 256 statistics), its vertical flip, near- and far-OOD (same idx as runs/feat/F_in256);
#   c256-*: covariate-shifted ID (hr_imagenet_{v2,c,r}, prep_hr_cs.py, all 1,000 images)
import os, json
import numpy as np
ROOT = os.path.expanduser('~/upknn'); F = f'{ROOT}/runs/feat/F_in256'; OUT = f'{ROOT}/runs/ok/sets_in256.json'
assert not os.path.exists(OUT), 'exists'
idx = lambda s: [int(i) for i in np.load(f'{F}/hr_{s}.npz')['idx']]
S = [['h256-imagenet_test', 'hr_imagenet_test', idx('imagenet_test'), 'none'],
     ['h256-imagenet_test__vflip', 'hr_imagenet_test', idx('imagenet_test'), 'vflip']]
for s in ['ssb_hard', 'ninco', 'inaturalist', 'textures', 'openimage_o']: S.append([f'h256-{s}', f'hr_{s}', idx(s), 'none'])
for k in ['v2', 'c', 'r']: S.append([f'c256-imagenet_{k}', f'hr_imagenet_{k}', list(range(1000)), 'none'])
json.dump(S, open(OUT, 'w')); print([(r[0], len(r[2])) for r in S])
