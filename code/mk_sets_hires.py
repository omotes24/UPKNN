# mk_sets_hires.py -- runs/ok/sets_in256c.json (prefix q256-; data hc_*) for PROTOCOL_HIRES.md.  The e512 sets (sets_e512.json, prefixes
#   q512- and rb512-) are written by enc_e512.py together with the latents.
import os, json
import numpy as np
ROOT = os.path.expanduser('~/upknn'); F = f'{ROOT}/runs/feat/F_in256'; OUT = f'{ROOT}/runs/ok/sets_in256c.json'
assert not os.path.exists(OUT), 'exists'
idx = lambda s: [int(i) for i in np.load(f'{F}/hr_{s}.npz')['idx']]
S = [['q256-imagenet_test', 'hc_imagenet_test', idx('imagenet_test'), 'none'], ['q256-imagenet_test__vflip', 'hc_imagenet_test', idx('imagenet_test'), 'vflip']]
for s in ['ssb_hard', 'ninco', 'inaturalist', 'textures', 'openimage_o']: S.append([f'q256-{s}', f'hc_{s}', idx(s), 'none'])
for k in ['v2', 'c', 'r']: S.append([f'q256-imagenet_{k}', f'hc_imagenet_{k}', list(range(1000)), 'none'])
json.dump(S, open(OUT, 'w')); print([(r[0], len(r[2])) for r in S])
