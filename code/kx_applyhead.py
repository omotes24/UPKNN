# kx_applyhead.py -- applies a trained rotation head (kx_rothead2.py output dir: heads.pt with the state dicts, mu, sd, T) to the stored
#   view-0 features of the sets with the given prefixes (e.g. final2-), writing <set>.npz (lp_*, lg0_*, H0, idx) like the trainer did.
#   No training happens here.   usage: python code/kx_applyhead.py --tag edmc10 --gpu 0 --seed 0 --k 0 --feat gq --tagout R --prefix final2-
import os, json, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ROOT = os.path.expanduser('~/upknn')
p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--gpu', type=int, default=0); p.add_argument('--seed', type=int, default=0)
p.add_argument('--k', type=int, default=0); p.add_argument('--feat', default='gq'); p.add_argument('--tagout', default='R'); p.add_argument('--prefix', required=True)
a = p.parse_args(); dev = torch.device(f'cuda:{a.gpu}')
od = f'{ROOT}/runs/kx/rothead/s{a.seed}/{a.tag}_k{a.k}_{a.feat}{a.tagout}'; ck = torch.load(f'{od}/heads.pt', map_location='cpu', weights_only=False)
SETS = [r[0] for r in json.load(open(f'{ROOT}/runs/ok/sets_{a.tag}.json')) if r[0].startswith(tuple(a.prefix.split(',')))]
def load(name):
    out = []
    for l in range(3):
        X = np.load(f'{ROOT}/runs/ok/s{a.seed}/{a.tag}/k{a.k}/{name}__L{l}.npy', mmap_mode='r'); C = X.shape[2] // 10
        out.append(np.asarray(X[:, :, :2 * C] if a.feat == 'g' else X[:, :, :10 * C]))
    return np.concatenate(out, 2)
mu, sd = ck['mu'].to(dev), ck['sd'].to(dev); D = mu.shape[0]; Hd = ck['mlp']['f1.weight'].shape[0]
class MLP(nn.Module):
    def __init__(s): super().__init__(); s.f1 = nn.Linear(D, Hd); s.f2 = nn.Linear(Hd, 4)
    def hid(s, x): return F.relu(s.f1(x))
    def forward(s, x): return s.f2(s.hid(x))
heads = {'lin': nn.Linear(D, 4), 'mlp': MLP()}
for hn, h in heads.items(): h.load_state_dict(ck[hn]); h.to(dev).eval()
def z(X): return (X.float() - mu) / sd
with torch.no_grad():
    for n in SETS:
        X = torch.from_numpy(load(n)).to(dev); out = {}
        for hn, h in heads.items():
            out[f'lp_{hn}'] = torch.stack([F.log_softmax(h(z(X[:, r])), 1)[:, r] for r in range(4)], 1).cpu().numpy()
            out[f'lg0_{hn}'] = h(z(X[:, 0])).float().cpu().numpy()
        out['H0'] = heads['mlp'].hid(z(X[:, 0])).half().cpu().numpy()
        np.savez(f'{od}/{n}.npz', **out, idx=np.load(f'{ROOT}/runs/ok/s{a.seed}/{a.tag}/k{a.k}/{n}__idx.npy'))
print('KX_APPLYHEAD_DONE', a.tag, a.seed, a.prefix, len(SETS), flush=True)
