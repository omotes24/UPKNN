# rotcls_eval.py -- scores of the three rotation classifiers of one ID (rotcls.py) on the common images of its pairs
#   bin90 = logit of the upright-vs-R270 classifier (estimate of Lambda_90), bin180 = logit of upright-vs-R180 (Lambda_180),
#   rot4 = sum_r log p(r | R_r x) of the four-way classifier, rot4_p0 = log p(0 | x).  Higher = more in-distribution.
#   usage: python eu/experiments/rotcls_eval.py <tag> <gpu>   -> runs/rotcls/<tag>/scores/<set>.npz
import os, sys, json
import numpy as np
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rotcls_net import build

ROOT = os.environ.get('SR_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
NPY = f'{ROOT}/data/npy'
ID = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'edmc100': 'cifar100_test',
      'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
tag, gpu = sys.argv[1], int(sys.argv[2])
sub = sys.argv[3] if len(sys.argv) > 3 else ''                      # optional variant directory (e.g. a second training seed)
dev = torch.device(f'cuda:{gpu}')
base = f'{ROOT}/runs/rotcls/{tag}{sub}'
nets = {}
for k in ('bin90', 'bin180', 'rot4'):
    ck = torch.load(f'{base}/{k}/final.pt', map_location='cpu', weights_only=False)
    n = build(ck['ch'], ck['n_out']); n.load_state_dict(ck['net']); nets[k] = n.to(dev).eval()
COM = json.load(open(os.environ.get('ROTCLS_COM', f'{ROOT}/eu/benchmark/common_idx.json')))[tag]
SCORES = os.environ.get('ROTCLS_SCORES', 'scores')
os.makedirs(f'{base}/{SCORES}', exist_ok=True)


@torch.no_grad()
def score(x):
    out = {}
    with torch.autocast('cuda', dtype=torch.float16):
        out['bin90'] = nets['bin90'](x).float()[:, 0]
        out['bin180'] = nets['bin180'](x).float()[:, 0]
        lp = [F.log_softmax(nets['rot4'](x.rot90(r, (2, 3)).contiguous()).float(), 1)[:, r] for r in range(4)]
    out['rot4'] = torch.stack(lp, 1).sum(1); out['rot4_p0'] = lp[0]
    return {k: v.double().cpu().numpy() for k, v in out.items()}


for s, idx in COM.items():
    name = ID[tag] if s == '__id__' else s
    data, vflip = (name[:-7], True) if name.endswith('__vflip') else (name, False)
    idx = np.array(sorted(int(i) for i in idx))
    xu = np.load(f'{NPY}/{data}.npy', mmap_mode='r')[idx]
    if xu.ndim == 3: xu = xu[..., None]
    res = {}
    for i in range(0, len(xu), 500):
        x = torch.from_numpy(np.ascontiguousarray(xu[i:i + 500])).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        if vflip: x = x.flip(2)
        for k, v in score(x).items(): res.setdefault(k, []).append(v)
    res = {k: np.concatenate(v) for k, v in res.items()}
    np.savez(f'{base}/{SCORES}/{name}.npz', idx=idx, **res)
    print(tag, name, len(idx), {k: round(float(v.mean()), 3) for k, v in res.items()}, flush=True)
print('EVAL_DONE', tag, flush=True)
