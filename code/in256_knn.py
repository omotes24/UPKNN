# in256_knn.py -- kNN (basic, PROTOCOL_CT) on the ImageNet-256 tier: as ood1k_knn.py (cosine of the L2-normalised [channel mean,
#   channel std] of the half-resolution decoder block L1 with the bank, 10th largest similarity), from the stored view-0 features.
#   usage: python code/in256_knn.py --seed 0 --gpu 0      -> runs/in256/knn_s<seed>.npz (ct__<set>, idx__<set>)
import os, json, argparse
import numpy as np, torch, torch.nn.functional as F
ROOT = os.path.expanduser('~/upknn')
p = argparse.ArgumentParser(); p.add_argument('--seed', type=int, default=0); p.add_argument('--gpu', type=int, default=0); a = p.parse_args()
dev = torch.device(f'cuda:{a.gpu}'); D = f'{ROOT}/runs/ok/s{a.seed}/in256/k0'
SETS = [r[0] for r in json.load(open(f'{ROOT}/runs/ok/sets_in256.json'))]
def load(name):
    X = np.load(f'{D}/{name}__L1.npy', mmap_mode='r'); C = X.shape[2] // 10
    return F.normalize(torch.from_numpy(np.ascontiguousarray(X[:, 0, :2 * C])).to(dev).float(), dim=1).half()
B = load('bank'); out = {}
with torch.no_grad():
    for n in SETS:
        Q = load(n)
        out[f'ct__{n}'] = np.concatenate([(Q[i:i + 1000] @ B.T).float().topk(10, dim=1).values[:, -1].cpu().numpy() for i in range(0, len(Q), 1000)]).astype(np.float64)
        out[f'idx__{n}'] = np.load(f'{D}/{n}__idx.npy')
os.makedirs(f'{ROOT}/runs/in256', exist_ok=True); np.savez(f'{ROOT}/runs/in256/knn_s{a.seed}.npz', **out)
print('IN256_KNN_DONE', a.seed, len(SETS), flush=True)
