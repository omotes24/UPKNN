# kx_probe.py -- DIAGNOSTIC ONLY (labels are never used by any detector): how much class information the stored pooled features carry
#   (linear probe, standardised features, 40k train / 10k test split of the 50k bank) and how much the cosine geometry exposes it
#   (10-NN majority vote with cosine).  usage: python code/kx_probe.py --tag edmc10 --gpu 0
import os, json, argparse, numpy as np, torch
ROOT = os.path.expanduser('~/upknn')
p = argparse.ArgumentParser(); p.add_argument('--tag', default='edmc10'); p.add_argument('--gpu', type=int, default=0); a = p.parse_args()
dev = torch.device(f'cuda:{a.gpu}')
YF = {'edmc10': 'cifar10_train_y', 'edmc100': 'cifar100_train_y'}[a.tag]
idx = np.load(f'{ROOT}/runs/ok/s0/{a.tag}/k0/bank__idx.npy'); y = torch.from_numpy(np.load(f'{ROOT}/data/npy/{YF}.npy')[idx].astype(np.int64)).to(dev)
ncls = int(y.max()) + 1
def feat(k, l, f):
    X = np.load(f'{ROOT}/runs/ok/s0/{a.tag}/k{k}/bank__L{l}.npy', mmap_mode='r'); C = X.shape[2] // 10
    s = {'g': np.r_[0:2 * C], 'm': np.r_[0:C], 's': np.r_[C:2 * C], 'q': np.r_[2 * C:10 * C], 'gq': np.r_[0:10 * C]}[f]
    return torch.from_numpy(np.ascontiguousarray(X[:, 0][:, s])).float().to(dev)
def probe(X):
    tr, te = slice(0, 40000), slice(40000, 50000)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Z = (X - mu) / sd
    W = torch.zeros(Z.shape[1], ncls, device=dev, requires_grad=True); b = torch.zeros(ncls, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=300, history_size=20, line_search_fn='strong_wolfe')
    def clo():
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(Z[tr] @ W + b, y[tr]) + 1e-4 * W.square().sum(); loss.backward(); return loss
    opt.step(clo)
    with torch.no_grad(): return float(((Z[te] @ W + b).argmax(1) == y[te]).float().mean())
def knnacc(X):
    Xn = torch.nn.functional.normalize(X, dim=1); tr, te = slice(0, 40000), slice(40000, 50000)
    S = Xn[te] @ Xn[tr].T; nn = S.topk(10, dim=1).indices; votes = y[tr][nn]
    pred = torch.mode(votes, dim=1).values; return float((pred == y[te]).float().mean())
out = {}
for k, l, f in [(0, 1, 'g'), (0, 1, 'm'), (0, 1, 's'), (0, 0, 'g'), (0, 2, 'g'), (0, 2, 'm'), (6, 1, 'g'), (6, 1, 'm'), (6, 2, 'g'), (6, 2, 'm'), (0, 1, 'gq'), (6, 2, 'gq')]:
    X = feat(k, l, f); r = (probe(X), knnacc(X)); out[f'L{l}.k{k}.{f}'] = r
    print(f'{a.tag} L{l} k{k} {f:3s} dim {X.shape[1]:5d}  linear {100 * r[0]:5.1f}  cos10NN {100 * r[1]:5.1f}', flush=True)
json.dump(out, open(f'{ROOT}/runs/kx/probe_{a.tag}.json', 'w'), indent=1)
