# vl_hooks3.py -- vl_hooks2.py in single precision (features kept in float32 on the CPU, cosines computed in float32 on the GPU; the
#   half-precision cosines of vl_hooks2.py / vl_eval.py round values near 1 and tie the nearest neighbours of the full-resolution level).
#   The mirror test as a measuring tool (development split and selection data only; never final2):
#   global [channel mean, channel std] of every encoder/decoder block (as vl_hooks.py) at one noise level (default sigma_0), for the bank
#   (ID training images) and the devall-/sel-/fresh- sets, plus the HORIZONTAL mirror images of the fresh ID images (control: the left-right
#   arrangement, which most ID sets do not fix).  For every block and rank R: the kNN content term -log r_R (cosine), its vertical mirror
#   test (fresh ID vs vertical flips), its horizontal mirror test (fresh ID vs horizontal flips), the selection-data AUROC and the
#   development-split AUROCs.  Also the three-level content space of Upright kNN (mean cosine of the last decoder block at 1, 1/2, 1/4
#   of the input resolution) and, for 64x64 inputs, the 1/8 level.
#   usage: python code/vl_hooks3.py --tag edmc10 --gpu 0 [--k 0]    output: runs/vl/hooks/<tag>_s0_k<k>_vh32.json
import os, sys, json, time, re, argparse
import numpy as np
import torch, torch.nn.functional as F
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgcore import ROOT, NPY, load_model, to_t, shared_noise, TF
from ct_extract_defs import SPEC, SIGS, PATS, part, bank_indices

p = argparse.ArgumentParser()
p.add_argument('--tag', required=True); p.add_argument('--gpu', type=int, default=0); p.add_argument('--k', type=int, default=0)
p.add_argument('--seed', type=int, default=0); p.add_argument('--hooks', default=r'(enc\.\d+x\d+_block\d+|dec\.\d+x\d+_(in\d+|block\d+))$|^(down_blocks\.\d+|mid_block|up_blocks\.\d+)$|^(input_blocks\.\d+|middle_block|output_blocks\.\d+)$')
p.add_argument('--R', default='1,2,5,10,50,200,1000')
a = p.parse_args()
spec, bankset, bs = SPEC[a.tag]
dev = torch.device(f'cuda:{a.gpu}'); torch.backends.cudnn.benchmark = True
den, net = load_model(spec, dev)
mods = dict(net.named_modules())
HOOKS = [n for n in mods if any(re.search(q, n) for q in PATS) and re.search(a.hooks, n)]
store, size = {}, {}


def grab(n):
    def f(m, inp, o):
        if isinstance(o, (tuple, list)): o = o[0]
        if torch.is_tensor(o) and o.ndim == 4:
            size[n] = int(o.shape[-1]); store[n] = torch.cat([o.float().mean((2, 3)), o.float().std((2, 3), correction=0)], 1)
    return f


for n in HOOKS: mods[n].register_forward_hook(grab(n))
SETS = [s for s in json.load(open(f'{ROOT}/runs/ok/sets_{a.tag}.json')) if s[0].startswith(('devall-', 'sel-', 'fresh-'))]
NAT = json.load(open(f'{ROOT}/code/kx_nat.json'))[a.tag]
idn = [s[0] for s in SETS if s[0].startswith('devall-')][0]; sel = [s[0] for s in SETS if s[0].startswith('sel-')]
fid = 'fresh-' + idn[7:]; vf = fid + '__vflip'; hf = fid + '__hflip'
vrow = [s for s in SETS if s[0] == vf][0]; SETS.append([hf, vrow[1], vrow[2], 'hflip'])   # the same images as the vertical mirror set
RS = [int(v) for v in a.R.split(',')]


def auroc(x, y):
    r = rankdata(np.concatenate([x, y])); return float((r[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))


@torch.no_grad()
def feats(xu, pre):
    out = {n: [] for n in HOOKS}; sig = SIGS[a.k]
    for i in range(0, len(xu), bs):
        x = to_t(np.ascontiguousarray(xu[i:i + bs]), dev)
        if pre != 'none': x = TF[pre][0](x)
        eps = shared_noise(x.shape[1:], a.seed, a.k, 0, dev); store.clear(); den(x + sig * eps, sig)
        for n in HOOKS: out[n].append(F.normalize(store[n].double(), dim=1).float().cpu())
    return {n: torch.cat(v) for n, v in out.items()}


t0 = time.time(); res = {'tag': a.tag, 'k': a.k, 'sigma': SIGS[a.k], 'hooks': HOOKS, 'rows': {}, 'precision': 'float32'}
full = np.load(f'{NPY}/{bankset}.npy', mmap_mode='r'); bidx, _ = bank_indices(len(full)); Rz = int(to_t(np.ascontiguousarray(full[:1]), dev).shape[-1])
Fb = feats(full[bidx], 'none'); Fs = {}
for name, data, idx, pre in SETS:
    Fs[name] = feats(np.load(f'{NPY}/{data}.npy', mmap_mode='r')[np.asarray(idx, dtype=np.int64)], pre)
print(f'{a.tag} k{a.k} features {time.time() - t0:.0f}s', flush=True)
dec = [n for n in HOOKS if part(n) == 'dec']
LEV = {}
for l in range(4):
    c = [n for n in dec if size.get(n) == (Rz >> l)]
    if c: LEV[l] = c[-1]                                              # the last decoder block at 1/2^l of the input resolution
res['levels'] = {f'L{l}': n for l, n in LEV.items()}; res['size'] = size


@torch.no_grad()
def evaluate(key, B, Q):
    B = B.to(dev).float()
    prof = {s: torch.cat([(Q[s][i:i + 1000].to(dev).float() @ B.T).topk(max(RS), dim=1).values[:, [r - 1 for r in RS]] for i in range(0, len(Q[s]), 1000)]).cpu().numpy().astype(np.float64) for s in Q}
    del B
    for j, R in enumerate(RS):
        S = {s: -np.log(np.sqrt(2 * np.clip(1 - prof[s][:, j], 1e-9, None))) for s in Q}
        fr = {o: 100 * auroc(S[fid], S['fresh-' + o]) for o in NAT}
        res['rows'][f'{key} R{R}'] = {'sel': float(np.mean([100 * auroc(S[idn], S[s]) for s in sel])), 'vflip': 100 * auroc(S[fid], S[vf]),
                                     'hflip': 100 * auroc(S[fid], S[hf]), 'nat': float(np.mean(list(fr.values()))), 'fresh': fr}


os.makedirs(f'{ROOT}/runs/vl/hooks', exist_ok=True); OUTF = f'{ROOT}/runs/vl/hooks/{a.tag}_s{a.seed}_k{a.k}_vh32.json'
for n in HOOKS:
    evaluate(n, Fb[n], {s: Fs[s][n] for s in Fs}); r1 = res['rows'][f'{n} R1']
    print(f'  {n:28s} {size.get(n)}: V {r1["vflip"]:5.1f} H {r1["hflip"]:5.1f} sel {r1["sel"]:5.1f} nat {r1["nat"]:6.2f}', flush=True)
json.dump(res, open(OUTF, 'w'), indent=0)                             # the per-block rows first, the level combinations below
print('levels', res['levels'], 'Rz', Rz, flush=True)
combos = {'L012': [0, 1, 2]} if all(l in LEV for l in (0, 1, 2)) else {}
if all(l in LEV for l in range(4)): combos['L0123'] = [0, 1, 2, 3]
for cn, ls in combos.items():
    cat = lambda D: torch.cat([D[LEV[l]].float() for l in ls], 1) / np.sqrt(len(ls))
    evaluate(cn, cat(Fb), {s: cat(Fs[s]) for s in Fs}); r1 = res['rows'][f'{cn} R1']
    print(f'  {cn:28s}: V {r1["vflip"]:5.1f} H {r1["hflip"]:5.1f} sel {r1["sel"]:5.1f} nat {r1["nat"]:6.2f}', flush=True)
json.dump(res, open(OUTF, 'w'), indent=0)
print('VL_HOOKS3_DONE', a.tag, f'{time.time() - t0:.0f}s', flush=True)
