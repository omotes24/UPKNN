# kx_ofk2.py -- DEVELOPMENT evaluation of the orientation-completed kNN (seed s):
#   score(x) = -d * log r_R(x) + log P_T(up | x)
#   r_R(x)^2 = 2 (1 - c_R(x)), c_R = R-th largest cosine of x's pooled decoder features (space h in {L1, L012}, sigma_1, view 0) over the
#   upright bank; (h, R) chosen on the selection data for the content term alone (as for the tuned kNN); d = Levina-Bickel intrinsic
#   dimension of the bank in that space (k = 20); P_T(up | x) = softmax(logits / T)_0 of the leak-free rotation head (kx_rothead2.py,
#   --tagout R; T fitted on held-out bank images).  Prints content, pose and the sum for every head variant; saves scores.
import os, sys, json, numpy as np, torch
ROOT = os.path.expanduser('~/upknn'); RS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
tag, gpu = sys.argv[1], int(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
dev = torch.device(f'cuda:{gpu}')
from scipy.stats import rankdata
def auroc(a, b):
    r = rankdata(np.concatenate([a, b])); return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))
ALL = [r[0] for r in json.load(open(f'{ROOT}/runs/ok/sets_{tag}.json'))]; SETS = [n for n in ALL if n.startswith(('devall-', 'sel-', 'fresh-'))]
NAT = json.load(open('code/kx_nat.json'))[tag]
idn = [n for n in SETS if n.startswith('devall-')][0]; sel = [n for n in SETS if n.startswith('sel-')]; fid = 'fresh-' + idn[7:]; vf = fid + '__vflip'
FR = [n for n in SETS if n.startswith('fresh-')]
def load(name, l):
    X = np.load(f'{ROOT}/runs/ok/s{seed}/{tag}/k0/{name}__L{l}.npy', mmap_mode='r'); C = X.shape[2] // 10
    return torch.nn.functional.normalize(torch.from_numpy(np.ascontiguousarray(X[:, 0, :2 * C])).to(dev).float(), dim=1)
HS = {'L1': [1], 'L012': [0, 1, 2]}
def space(name, h): return (torch.cat([load(name, l) for l in HS[h]], 1) / np.sqrt(len(HS[h]))).half()
def lid(B, k=20, m=3000):
    idx = torch.from_numpy(np.random.RandomState(0).choice(len(B), min(m, len(B)), replace=False)).to(dev)
    c = (B[idx] @ B.T).float().topk(k + 1, dim=1).values[:, 1:].clamp(max=1 - 1e-6); r = torch.sqrt(2 * (1 - c))
    return float((1 / torch.log(r[:, -1:] / r[:, :-1]).mean(1)).mean())
best = None
for h in HS:
    B = space('bank', h); prof = {}
    with torch.no_grad():
        for n in SETS:
            Q = space(n, h); prof[n] = np.concatenate([(Q[i:i + 1000] @ B.T).float().topk(1000, dim=1).values[:, [r - 1 for r in RS]].cpu().numpy() for i in range(0, len(Q), 1000)]).astype(np.float64)
    for j, R in enumerate(RS):
        sm = np.mean([100 * auroc(prof[idn][:, j], prof[s][:, j]) for s in sel])
        if best is None or sm > best[0] + 1e-9: best = (sm, h, j, R, {n: prof[n][:, j].copy() for n in SETS}, lid(B))
    del B; torch.cuda.empty_cache()
sm, h, j, R, cR, d = best
content = {n: -d * 0.5 * np.log(2 * (1 - np.minimum(cR[n], 1 - 1e-9))) for n in SETS}
def logsm(lg, T): z = lg / T; return z[:, 0] - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1)) - z.max(1)
POSE = {}
for to in os.environ.get('KX_TAGOUTS', 'R').split(','):
  for f in ('gq', 'g'):
    src = f'{ROOT}/runs/kx/rothead/s{seed}/{tag}_k0_{f}{to}'
    if not os.path.exists(f'{src}/temp.json'): continue
    T = json.load(open(f'{src}/temp.json'))
    for hn in ('lin', 'mlp'): POSE[f'{f}{to}.{hn}' + (f' nll={T[hn + "_valnll"]:.3f}' if hn + '_valnll' in T else '')] = {n: logsm(np.load(f'{src}/{n}.npz')[f'lg0_{hn}'].astype(np.float64), T[hn]) for n in SETS}
res = {'tag': tag, 'seed': seed, 'content': {'space': h, 'R': R, 'd': d, 'sel': sm}, 'rows': {}}
def rep(name, S):
    s_ = float(np.mean([100 * auroc(S[idn], S[s]) for s in sel])); fr = {o: 100 * auroc(S[fid], S['fresh-' + o]) for o in NAT}
    v = 100 * auroc(S[fid], S[vf]); res['rows'][name] = {'sel': s_, 'fresh': fr, 'nat': float(np.mean(list(fr.values()))), 'vflip': v}
    print(f'{tag:8s} s{seed} {name:30s} sel {s_:6.2f}  fresh nat {np.mean(list(fr.values())):6.2f}  vflip {v:5.1f}   ' + ' '.join(f'{x:5.1f}' for x in fr.values()), flush=True)
rep(f'content {h} P{R} d={d:.1f}', content)
for k, P in POSE.items(): rep(f'pose {k}', P)
for k, P in POSE.items(): rep(f'content+pose {k}', {n: content[n] + P[n] for n in SETS})
os.makedirs(f'{ROOT}/runs/kx/ofk2', exist_ok=True)
json.dump(res, open(f'{ROOT}/runs/kx/ofk2/{tag}_s{seed}' + os.environ.get('KX_SUFFIX', '') + '.json', 'w'), indent=1)
np.savez(f'{ROOT}/runs/kx/ofk2/{tag}_s{seed}' + os.environ.get('KX_SUFFIX', '') + '_scores.npz', **{f'content__{n}': content[n] for n in FR + [idn] + sel}, **{f'pose_{k}__{n}': P[n] for k, P in POSE.items() for n in FR + [idn] + sel})
