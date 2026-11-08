# vl_eval.py -- Vertical Limit exploration on the features of vl_extract.py (seed s, noise level k): for every content space
#   (L2, L1+L2, L012; cosine of the L2-normalised [mean, std] features, several blocks = the mean of their cosines) and rank R, the
#   kNN term -d log r_R with the Levina-Bickel d of the bank in that space (k = 20, 3000 points); the rotation head of kx_rothead2.py
#   (MLP D-512-4 on [L0, L1, L2] x gq of the four rotated bank views, noise not rotated, 5000 held-out bank images, temperature by
#   held-out NLL) and its held-out NLL / accuracy (ID-only); AUROCs on the development split (fresh-) and on the selection data
#   (devall ID vs sel-).  final2 is never read.
#   usage: python code/vl_eval.py --tag edmc10 --gpu 0 --k 2 [--seed 0] [--bank bank]
#   output: runs/vl/eval/<tag>_s<seed>_k<k>.json (+ _scores.npz with the per-image terms)
import os, sys, json, time, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/upknn')
p = argparse.ArgumentParser()
p.add_argument('--tag', required=True); p.add_argument('--gpu', type=int, default=0); p.add_argument('--k', type=int, required=True)
p.add_argument('--seed', type=int, default=0); p.add_argument('--bank', default='bank'); p.add_argument('--epochs', type=int, default=8); p.add_argument('--levels', default='0,1,2'); p.add_argument('--headbank', default='bank'); p.add_argument('--rs', default='1,2,5,10,20,50,100,200,500,1000')
p.add_argument('--spaces', default=''); p.add_argument('--outtag', default='')
a = p.parse_args()
dev = torch.device(f'cuda:{a.gpu}'); torch.manual_seed(0)
FD = f'{ROOT}/runs/vl/feat/s{a.seed}/{a.tag}/k{a.k}'
SETS = [r[0] for r in json.load(open(f'{ROOT}/runs/ok/sets_{a.tag}.json')) if r[0].startswith(('devall-', 'sel-', 'fresh-'))]
NAT = json.load(open(f'{ROOT}/code/kx_nat.json'))[a.tag]
idn = [n for n in SETS if n.startswith('devall-')][0]; sel = [n for n in SETS if n.startswith('sel-')]
fid = 'fresh-' + idn[7:]; vf = fid + '__vflip'
RS = [int(v) for v in a.rs.split(',')]
SPACES = {'L0': ([0], 'g', False), 'L1': ([1], 'g', False), 'L2': ([2], 'g', False), 'L1+L2': ([1, 2], 'g', False), 'L012': ([0, 1, 2], 'g', False), 'L2z': ([2], 'g', True),
          'L012z': ([0, 1, 2], 'g', True), 'L2qz': ([2], 'gq', True), 'L012qz': ([0, 1, 2], 'gq', True), 'L2q': ([2], 'gq', False),
          'L2m': ([2], 'm', False), 'L2qm': ([2], 'qm', False)}
LEVELS = [int(v) for v in a.levels.split(',')]
if 3 in LEVELS: SPACES.update({'L3': ([3], 'g', False), 'L3z': ([3], 'g', True), 'L23z': ([2, 3], 'g', True), 'L0123z': ([0, 1, 2, 3], 'g', True),
                               'L3q': ([3], 'gq', False), 'L3m': ([3], 'm', False), 'L3qm': ([3], 'qm', False)})
if a.spaces: SPACES = {k_: v_ for k_, v_ in SPACES.items() if k_ in a.spaces.split(',')}
t0 = time.time()


def auroc(x, y):
    r = rankdata(np.concatenate([x, y])); return float((r[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))


STATS = {}


def loadg(name, l, part='g', zs=False):
    allv = name.endswith('@4'); name = name[:-2] if allv else name           # '<bank>@4': the four rotated views as separate entries
    X = np.load(f'{FD}/{name}__L{l}.npy', mmap_mode='r'); C = X.shape[-1] // 10; w = 2 * C if part == 'g' else 10 * C
    cols = {'g': slice(0, 2 * C), 'gq': slice(0, 10 * C), 'm': slice(0, C), 'qm': slice(2 * C, 6 * C)}[part]   # m: channel means only; qm: 2x2 grid means
    if allv: Y = torch.from_numpy(np.ascontiguousarray(X[:, :, cols])).to(dev).float().reshape(-1, cols.stop - cols.start)
    else: Y = torch.from_numpy(np.ascontiguousarray(X[:, 0, cols] if X.ndim == 3 else X[:, cols])).to(dev).float()
    if zs:                                                            # per-dimension standardisation with the bank (ID-only)
        if (l, part) not in STATS:
            Bk = np.load(f'{FD}/{a.bank.split("+")[0]}__L{l}.npy', mmap_mode='r'); Bk = torch.from_numpy(np.ascontiguousarray(Bk[:, 0, cols] if Bk.ndim == 3 else Bk[:, cols])).to(dev).float()
            STATS[(l, part)] = (Bk.mean(0), Bk.std(0) + 1e-3); del Bk
        m_, s_ = STATS[(l, part)]; Y = (Y - m_) / s_
    return F.normalize(Y, dim=1)


def space(name, ls, part='g', zs=False): return (torch.cat([loadg(name, l, part, zs) for l in ls], 1) / np.sqrt(len(ls))).half()


def lid(B, k=20, m=3000):
    idx = torch.from_numpy(np.random.RandomState(0).choice(len(B), min(m, len(B)), replace=False)).to(dev)
    c = torch.cat([(B[idx[i:i + 500]] @ B.T).float().topk(k + 1, dim=1).values[:, 1:] for i in range(0, len(idx), 500)]).clamp(max=1 - 1e-6); r = torch.sqrt(2 * (1 - c))
    return float((1 / torch.log(r[:, -1:] / r[:, :-1]).mean(1)).mean())


res = {'tag': a.tag, 'seed': a.seed, 'k': a.k, 'content': {}, 'head': {}}
content = {}                                                          # (space, R) -> {set: -d log r_R}
for sn, (ls, part, zs) in SPACES.items():
    B = torch.cat([space(b_, ls, part, zs) for b_ in a.bank.split('+')]); d = lid(B); res['content'][sn] = {'d': d, 'N': len(B)}
    prof = {}
    with torch.no_grad():
        for n in SETS:
            Q = space(n, ls, part, zs)
            prof[n] = np.concatenate([(Q[i:i + 500] @ B.T).float().topk(max(RS), dim=1).values[:, [r - 1 for r in RS]].cpu().numpy() for i in range(0, len(Q), 500)]).astype(np.float64)
    for j, R in enumerate(RS):
        content[(sn, R)] = {n: -d * 0.5 * np.log(2 * (1 - np.minimum(prof[n][:, j], 1 - 1e-9))) for n in SETS}
    del B; torch.cuda.empty_cache(); print(a.tag, 'k', a.k, sn, 'd', round(d, 2), f'{time.time() - t0:.0f}s', flush=True)

# ---- the rotation head (as kx_rothead2.py, feat gq on L0, L1, L2)
def loadq(name):
    if '+' in name: return np.concatenate([loadq(n_) for n_ in name.split('+')], 0)
    out = []
    for l in range(3):
        X = np.load(f'{FD}/{name}__L{l}.npy', mmap_mode='r'); out.append(np.asarray(X))
    return np.concatenate(out, -1)


Xb = torch.from_numpy(loadq(a.headbank)).to(dev)                          # [Nb, 4, D] half
Nb, _, D = Xb.shape
s1 = torch.zeros(D, device=dev, dtype=torch.float64); s2 = torch.zeros(D, device=dev, dtype=torch.float64)
for i in range(0, Nb, 2000):
    xc = Xb[i:i + 2000].double().reshape(-1, D); s1 += xc.sum(0); s2 += xc.square().sum(0)
cnt = 4 * Nb; mu = (s1 / cnt).float(); sd = ((s2 / cnt - (s1 / cnt) ** 2).clamp_min(0) * cnt / (cnt - 1)).sqrt().float() + 1e-3; del s1, s2
nparts = len(a.headbank.split('+')); N0 = Nb // nparts                # extra parts (mirror images) are aligned with the first bank
perm = torch.from_numpy(np.random.RandomState(0).permutation(N0)).to(dev); va = perm[:5000]
tr = torch.cat([perm[5000:] + j * N0 for j in range(nparts)])          # the mirror images of the held-out images stay out of training


def z(X): return (X.float() - mu) / sd


class MLP(nn.Module):
    def __init__(s): super().__init__(); s.f1 = nn.Linear(D, 512); s.f2 = nn.Linear(512, 4)
    def forward(s, x): return s.f2(F.relu(s.f1(x)))


h = MLP().to(dev); opt = torch.optim.AdamW(h.parameters(), lr=1e-3, weight_decay=1e-4); bs = 512
steps = a.epochs * (len(tr) // bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
for ep in range(a.epochs):
    pe = tr[torch.randperm(len(tr), device=dev)]
    for i in range(0, len(pe) - bs + 1, bs):
        b = pe[i:i + bs]; x = z(Xb[b]).reshape(-1, D); y = torch.arange(4, device=dev).repeat(len(b))
        loss = F.cross_entropy(h(x), y); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
h.eval()
with torch.no_grad():
    xv = z(Xb[va]).reshape(-1, D); yv = torch.arange(4, device=dev).repeat(len(va)); lg = h(xv).float()
    acc = float((lg.argmax(1) == yv).float().mean()); Ts = torch.exp(torch.linspace(np.log(0.25), np.log(8), 200, device=dev))
    nll = torch.stack([F.cross_entropy(lg / T, yv) for T in Ts]); T = float(Ts[nll.argmin()])
res['head'] = {'acc': acc, 'nll': float(nll.min()), 'T': T}
print(a.tag, 'k', a.k, 'head acc', round(100 * acc, 2), 'nll', round(float(nll.min()), 4), 'T', round(T, 3), f'{time.time() - t0:.0f}s', flush=True)
del Xb; torch.cuda.empty_cache()
pose = {}
with torch.no_grad():
    for n in SETS:
        Xs = torch.from_numpy(loadq(n)).to(dev); pose[n] = F.log_softmax(h(z(Xs)).double() / T, 1)[:, 0].cpu().numpy()


def rep(S):
    s_ = float(np.mean([100 * auroc(S[idn], S[s]) for s in sel])); fr = {o: 100 * auroc(S[fid], S['fresh-' + o]) for o in NAT}
    return {'sel': s_, 'nat': float(np.mean(list(fr.values()))), 'vflip': 100 * auroc(S[fid], S[vf]), 'fresh': fr}


res['rows'] = {'pose': rep(pose)}
for (sn, R), C in content.items():
    res['rows'][f'content {sn} R{R}'] = rep(C)
    res['rows'][f'sum {sn} R{R}'] = rep({n: C[n] + pose[n] for n in SETS})
os.makedirs(f'{ROOT}/runs/vl/eval', exist_ok=True)
OUT = f'{ROOT}/runs/vl/eval/{a.tag}_s{a.seed}_k{a.k}{"" if a.bank == "bank" else "_" + a.bank}{"_" + a.outtag if a.outtag else ""}'
json.dump(res, open(OUT + '.json', 'w'), indent=1)
np.savez(OUT + '_scores.npz', **{f'pose__{n}': pose[n] for n in SETS},
         **{f'content_{sn}_R{R}__{n}': C[n] for (sn, R), C in content.items() for n in SETS})
best = sorted(res['rows'].items(), key=lambda kv: -kv[1]['nat'])[:5]
print(a.tag, 'k', a.k, 'top fresh nat:', [(k_, round(v['nat'], 2), round(v['sel'], 2), round(v['vflip'], 1)) for k_, v in best], flush=True)
print('VL_EVAL_DONE', a.tag, a.k, f'{time.time() - t0:.0f}s', flush=True)
