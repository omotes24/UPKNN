# knn_basic.py -- kNN (basic, PROTOCOL_CT) for any tier: L1 block, L2-normalised [channel mean, channel std] of view 0, cosine with the
#   bank, 10th largest similarity.   usage: python code/knn_basic.py --tag in256c --seed 0 --gpu 0 --prefix q256-  -> runs/hires/knn_<tag>_s<seed>.npz
import os, json, argparse
import numpy as np, torch, torch.nn.functional as F
ROOT = os.path.expanduser('~/upknn')
p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--seed', type=int, default=0); p.add_argument('--gpu', type=int, default=0)
p.add_argument('--prefix', required=True); a = p.parse_args()
dev = torch.device(f'cuda:{a.gpu}'); D = f'{ROOT}/runs/ok/s{a.seed}/{a.tag}/k0'
SETS = [r[0] for r in json.load(open(f'{ROOT}/runs/ok/sets_{a.tag}.json')) if r[0].startswith(a.prefix)]
def load(name):
    X = np.load(f'{D}/{name}__L1.npy', mmap_mode='r'); C = X.shape[2] // 10
    return F.normalize(torch.from_numpy(np.ascontiguousarray(X[:, 0, :2 * C])).to(dev).float(), dim=1).half()
B = load('bank'); out = {}
with torch.no_grad():
    for n in SETS:
        Q = load(n)
        out[f'ct__{n}'] = np.concatenate([(Q[i:i + 1000] @ B.T).float().topk(10, dim=1).values[:, -1].cpu().numpy() for i in range(0, len(Q), 1000)]).astype(np.float64)
        out[f'idx__{n}'] = np.load(f'{D}/{n}__idx.npy')
os.makedirs(f'{ROOT}/runs/hires', exist_ok=True); np.savez(f'{ROOT}/runs/hires/knn_{a.tag}_s{a.seed}.npz', **out)
print('KNN_BASIC_DONE', a.tag, a.seed, len(SETS), flush=True)
