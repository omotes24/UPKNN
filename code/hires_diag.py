# hires_diag.py -- POST-HOC DIAGNOSTICS of the resolution-controlled tiers (no change to the method or to any result):
#   (1) in the space of the kNN term (unit-length [channel mean, std] per block), for each tier: the share of the squared norm carried by
#       the largest channel of each block (median over bank images), and the cosine similarity of random pairs of bank images (quantiles),
#       to explain the degenerate neighbourhoods of dit256 in the d estimate;
#   (2) a pixel-statistics control for the bottleneck: per 224-px image (hc224_<set>, the evaluation images of in256c), the mean absolute
#       Laplacian (high-frequency energy) and the mean absolute horizontal difference across 8-pixel column boundaries minus within columns
#       (JPEG blockiness); AUROC of ID (ImageNet val) against every OOD and csID set, either direction reported as max(A, 1 - A).
#   usage: python code/hires_diag.py   -> runs/hires/hires_diag.json
import os, json, numpy as np
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/upknn')
def auroc(a, b): r = rankdata(np.concatenate([a, b])); return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))
out = {'feat': {}, 'pix': {}}
rs = np.random.RandomState(0)
for tag in ('in256c', 'e512', 'dit256'):
    B = [np.load(f'{ROOT}/runs/kx/rotb/s0/{tag}/k0/bank__L{l}.npy', mmap_mode='r') for l in range(3)]
    sub = np.sort(rs.choice(B[0].shape[0], 2000, replace=False)); res = {}
    G = []
    for l, b in enumerate(B):
        C = b.shape[2] // 10; x = np.asarray(b[sub, 0, :2 * C], np.float64)
        sq = x ** 2; top = sq.max(1) / sq.sum(1); res[f'L{l}_top_channel_share_median'] = float(np.median(top))
        res[f'L{l}_argmax_channel_mode_frac'] = float(np.bincount(sq.argmax(1)).max() / len(sub))
        G.append(x / np.linalg.norm(x, axis=1, keepdims=True))
    g = np.concatenate(G, 1) / np.sqrt(3)
    i, j = rs.randint(0, len(sub), 20000), rs.randint(0, len(sub), 20000); k = i != j
    cs = (g[i[k]] * g[j[k]]).sum(1); res['pair_cos_quantiles_1_50_99'] = [float(np.quantile(cs, q)) for q in (0.01, 0.5, 0.99)]
    res['frac_pairs_cos_ge_0.9999'] = float(np.mean(cs >= 0.9999))
    out['feat'][tag] = res; print(tag, res, flush=True)
SETS = json.load(open(f'{ROOT}/runs/ok/sets_in256c.json'))
def pixstats(name, idx):
    a = np.load(f'{ROOT}/data/npy/hc224_{name}.npy', mmap_mode='r')
    hf, bk = [], []
    for t in idx:
        x = np.asarray(a[int(t)], np.float64).mean(2)                     # grey
        lap = np.abs(4 * x[1:-1, 1:-1] - x[:-2, 1:-1] - x[2:, 1:-1] - x[1:-1, :-2] - x[1:-1, 2:]); hf.append(lap.mean())
        dh = np.abs(np.diff(x, axis=1)); cols = np.arange(dh.shape[1]); on = (cols % 8) == 7
        bk.append(dh[:, on].mean() - dh[:, ~on].mean())
    return np.array(hf), np.array(bk)
rows = {r[0].replace('q256-', ''): (r[1], r[2]) for r in SETS if r[0].startswith('q256-') and r[3] == 'none'}
print('pixel sets', list(rows), flush=True)
idn = 'imagenet_test'; arr_of = lambda nm: nm.replace('hc_', '')
hf0, bk0 = pixstats(arr_of(rows[idn][0]), rows[idn][1])
for s, (arr, idx) in rows.items():
    if s == idn: continue
    hf, bk = pixstats(arr_of(arr), idx); a1, a2 = auroc(hf0, hf), auroc(bk0, bk)
    out['pix'][s] = dict(hf_auroc=100 * max(a1, 1 - a1), hf_dir='ID higher' if a1 > 0.5 else 'OOD higher', blk_auroc=100 * max(a2, 1 - a2),
                         blk_dir='ID higher' if a2 > 0.5 else 'OOD higher', hf_med=[float(np.median(hf0)), float(np.median(hf))], n=len(idx))
    print(s, out['pix'][s], flush=True)
json.dump(out, open(f'{ROOT}/runs/hires/hires_diag.json', 'w'), indent=1)
