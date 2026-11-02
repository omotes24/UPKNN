# kx_rothead.py -- ID-only training: a rotation-prediction head on the FROZEN pooled decoder features of runs/ok (all four rotated
#   views of the bank are stored).  Input: [L0, L1, L2] x (g | gq) of view r at noise level k (standardised on the bank);
#   target: r.  Heads: linear and MLP (D-512-4).  Hold-out: 5000 bank images (fixed permutation, seed 0) for the accuracy only.
#   Saves, for every set with the given prefixes, lp [N, 4] = log p(r | view r) and the MLP hidden code of view 0 (H0, 512, fp16),
#   plus the bank's H0.   usage: python code/kx_rothead.py --tag edmc10 --gpu 0 [--k 0] [--feat gq] [--prefix devall-,sel-,fresh-]
import os, json, time, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.expanduser('~/upknn')
p = argparse.ArgumentParser()
p.add_argument('--tag', required=True); p.add_argument('--gpu', type=int, default=0); p.add_argument('--k', type=int, default=0)
p.add_argument('--feat', default='gq'); p.add_argument('--seed', type=int, default=0); p.add_argument('--prefix', default='devall-,sel-,fresh-')
p.add_argument('--epochs', type=int, default=8); p.add_argument('--bankdir', default='runs/ok'); p.add_argument('--tagout', default='')
a = p.parse_args()
dev = torch.device(f'cuda:{a.gpu}'); torch.manual_seed(0)
SETS = [r[0] for r in json.load(open(f'{ROOT}/runs/ok/sets_{a.tag}.json')) if r[0].startswith(tuple(a.prefix.split(',')))]
od = f'{ROOT}/runs/kx/rothead/s{a.seed}/{a.tag}_k{a.k}_{a.feat}{a.tagout}'; os.makedirs(od, exist_ok=True)


def load(name):
    out = []
    for l in range(3):
        X = np.load(f'{ROOT}/{a.bankdir if name == "bank" else "runs/ok"}/s{a.seed}/{a.tag}/k{a.k}/{name}__L{l}.npy', mmap_mode='r'); C = X.shape[2] // 10
        out.append(np.asarray(X[:, :, :2 * C] if a.feat == 'g' else X[:, :, :10 * C]))
    return np.concatenate(out, 2)                                   # [N, 4, D] float16


t0 = time.time()
Xb = torch.from_numpy(load('bank')).to(dev)                          # [Nb, 4, D] half
Nb, _, D = Xb.shape
mu = Xb.float().mean((0, 1)); sd = Xb.float().std((0, 1)) + 1e-3
perm = torch.from_numpy(np.random.RandomState(0).permutation(Nb)).to(dev); va, tr = perm[:5000], perm[5000:]
print(a.tag, 'bank', tuple(Xb.shape), f'{time.time() - t0:.0f}s', flush=True)


def z(X): return (X.float() - mu) / sd


class MLP(nn.Module):
    def __init__(s): super().__init__(); s.f1 = nn.Linear(D, 512); s.f2 = nn.Linear(512, 4)
    def hid(s, x): return F.relu(s.f1(x))
    def forward(s, x): return s.f2(s.hid(x))


heads = {'lin': nn.Linear(D, 4).to(dev), 'mlp': MLP().to(dev)}
for hn, h in heads.items():
    opt = torch.optim.AdamW(h.parameters(), lr=1e-3, weight_decay=1e-4); bs = 512
    steps = a.epochs * (len(tr) // bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    it = 0
    for ep in range(a.epochs):
        pe = tr[torch.randperm(len(tr), device=dev)]
        for i in range(0, len(pe) - bs + 1, bs):
            b = pe[i:i + bs]; x = z(Xb[b]).reshape(-1, D); y = torch.arange(4, device=dev).repeat(len(b))
            loss = F.cross_entropy(h(x), y); opt.zero_grad(); loss.backward(); opt.step(); sch.step(); it += 1
        with torch.no_grad():
            xv = z(Xb[va]).reshape(-1, D); yv = torch.arange(4, device=dev).repeat(len(va)); acc = (h(xv).argmax(1) == yv).float().mean().item()
        print(f'{a.tag} {hn} epoch {ep} loss {loss.item():.3f} val acc {100 * acc:.1f} ({time.time() - t0:.0f}s)', flush=True)
    h.eval()


TEMP = {}
with torch.no_grad():                                               # temperature scaling on the held-out bank images (ID only)
    xv = z(Xb[va]).reshape(-1, D); yv = torch.arange(4, device=dev).repeat(len(va))
    for hn, h in heads.items():
        lg = h(xv).float(); Ts = torch.exp(torch.linspace(np.log(0.25), np.log(8), 200, device=dev))
        nll = torch.stack([F.cross_entropy(lg / T, yv) for T in Ts]); TEMP[hn] = float(Ts[nll.argmin()]); TEMP[hn + "_valnll"] = float(nll.min())
print(a.tag, 'temperatures', TEMP, flush=True)


@torch.no_grad()
def run(X):
    out = {}
    for hn, h in heads.items():
        lp = torch.stack([F.log_softmax(h(z(X[:, r])), 1)[:, r] for r in range(4)], 1); out[f'lp_{hn}'] = lp.cpu().numpy()
        out[f'lg0_{hn}'] = h(z(X[:, 0])).float().cpu().numpy()
    out['H0'] = heads['mlp'].hid(z(X[:, 0])).half().cpu().numpy(); return out


R = {}
for i in range(0, Nb, 5000):
    o = run(Xb[i:i + 5000]); R.setdefault('H0', []).append(o['H0'])
np.save(f'{od}/bank__H0.npy', np.concatenate(R['H0']))
del Xb; torch.cuda.empty_cache()
for n in SETS:
    X = torch.from_numpy(load(n)).to(dev); o = run(X)
    np.savez(f'{od}/{n}.npz', **o, idx=np.load(f'{ROOT}/runs/ok/s{a.seed}/{a.tag}/k{a.k}/{n}__idx.npy'))
torch.save({hn: h.state_dict() for hn, h in heads.items()} | {'mu': mu.cpu(), 'sd': sd.cpu(), 'T': TEMP}, f'{od}/heads.pt'); json.dump(TEMP, open(f'{od}/temp.json', 'w'))
print('KX_ROTHEAD_DONE', a.tag, f'{time.time() - t0:.0f}s', flush=True)
