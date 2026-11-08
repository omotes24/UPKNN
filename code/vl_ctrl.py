# vl_ctrl.py -- the two controls of the Vertical-Limit section (development split; the choices by the selection data; final2 never read).
#  (1) conditional kNN (chain rule inside one density estimate): A = the deep level l* of VL at sigma_0 ([channel mean, std]),
#      B = the content space of Upright kNN (the last decoder blocks at 1, 1/2, 1/4 of the resolution, mean cosine).
#      c_{B|A}(x) = -d_B log r, r = sqrt(2 (1 - c)), c = the R_c-th largest B-cosine among the K bank images nearest to x in A
#      (R_c = min(R_U, K // 5)); chain score = c_A + c_{B|A} + log q (the density of (A, B) by the chain rule, the pose term once).
#      K in {50, 200, 1000, 5000}: the largest selection-data AUROC of the chain score (noise seed 0).
#  (2) a channel-mean expert: c_m = -d_m log r_{R_m} on the channel means of l* at sigma_0 (the first C entries of [mean, std]);
#      score = VL + (c_m + log q).  R_m: the largest selection-data AUROC of c_m alone (the rule of R* of VL), noise seed 0.
#  Development split, noise seeds 0-2: the same images and order as runs/kx/export (Upright) and runs/vl/export (VL).
#  output: runs/vl/ctrl/<tag>_s<seed>_fresh.npz (cA, cBA, cB, cM, idx per fresh set), runs/vl/ctrl/<tag>_choice.json
#  usage: python code/vl_ctrl.py --tag edmc10 --gpu 0
import os, json, time, argparse
import numpy as np
import torch, torch.nn.functional as F
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/upknn')
p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--gpu', type=int, default=0)
p.add_argument('--Ks', default='50,200,1000,5000'); p.add_argument('--Rm', default='1,2,5,10,20,50,100,200,500,1000')
a = p.parse_args(); tag = a.tag
dev = torch.device(f'cuda:{a.gpu}'); t0 = time.time()
P = json.load(open(f'{ROOT}/runs/vl/PROTOCOL_VL.json'))[tag]; lv, Rs, RU = P['level'], P['R'], P['R_upright']
KS = [int(v) for v in a.Ks.split(',')]; RM = [int(v) for v in a.Rm.split(',')]
SETS = [r[0] for r in json.load(open(f'{ROOT}/runs/ok/sets_{tag}.json'))]
idn = [n for n in SETS if n.startswith('devall-')][0]; sel = [n for n in SETS if n.startswith('sel-')]
FRESH = [n for n in SETS if n.startswith('fresh-')]


def auroc(x, y):
    r = rankdata(np.concatenate([x, y])); return 100 * float((r[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))


def cols(X, part, tenC):                                              # runs/vl/feat and runs/ok store 10C columns ([mean, std] first), featf 2C
    C = X.shape[-1] // (10 if tenC else 2)
    return {'g': slice(0, 2 * C), 'm': slice(0, C)}[part]


def load(path, part, tenC):
    X = np.load(path, mmap_mode='r'); X = X[:, 0] if X.ndim == 3 else X   # the upright view
    return F.normalize(torch.from_numpy(np.ascontiguousarray(X[:, cols(X, part, tenC)])).to(dev).float(), dim=1)


def space(paths, part, tenC): return (torch.cat([load(q, part, tenC) for q in paths], 1) / np.sqrt(len(paths))).half()


def lid(B, k=20, m=3000):
    idx = torch.from_numpy(np.random.RandomState(0).choice(len(B), min(m, len(B)), replace=False)).to(dev)
    c = torch.cat([(B[idx[i:i + 500]] @ B.T).float().topk(k + 1, dim=1).values[:, 1:] for i in range(0, len(idx), 500)]).clamp(max=1 - 1e-6)
    r = torch.sqrt(2 * (1 - c)); return float((1 / torch.log(r[:, -1:] / r[:, :-1]).mean(1)).mean())


def nl(c): return -0.5 * np.log(2 * (1 - np.minimum(c, 1 - 1e-9)))     # -log r


@torch.no_grad()
def kterm(Q, B, Rlist):
    return np.concatenate([(Q[i:i + 500] @ B.T).float().topk(max(Rlist), dim=1).values[:, [r - 1 for r in Rlist]].cpu().numpy() for i in range(0, len(Q), 500)]).astype(np.float64)


@torch.no_grad()
def cond(QA, QB, BA, BB, Ks):
    out = {K: [] for K in Ks}; bs = max(1, int(1.5e8 / (max(Ks) * BB.shape[1])))
    for i in range(0, len(QA), bs):
        top = (QA[i:i + bs] @ BA.T).float().topk(max(Ks), dim=1).indices
        sB = torch.einsum('bd,bkd->bk', QB[i:i + bs].float(), BB[top].float())
        for K in Ks: out[K].append(sB[:, :K].topk(max(1, min(RU, K // 5)), dim=1).values[:, -1].cpu().numpy())
    return {K: np.concatenate(v).astype(np.float64) for K, v in out.items()}


# ---- choices on the selection data (noise seed 0; features of vl_extract.py, pose of vl_eval.py)
FD = f'{ROOT}/runs/vl/feat/s0/{tag}/k0'
PZ = np.load(f'{ROOT}/runs/vl/eval/{tag}_s0_k0_full_scores.npz')
pick = [idn] + sel
BA = space([f'{FD}/bank__L{lv}.npy'], 'g', True); BB = space([f'{FD}/bank__L{l}.npy' for l in (0, 1, 2)], 'g', True); BM = space([f'{FD}/bank__L{lv}.npy'], 'm', True)
dA, dB, dM = lid(BA), lid(BB), lid(BM)
cA, cBA, cM = {}, {}, {}
for n in pick:
    QA = space([f'{FD}/{n}__L{lv}.npy'], 'g', True); QB = space([f'{FD}/{n}__L{l}.npy' for l in (0, 1, 2)], 'g', True); QM = space([f'{FD}/{n}__L{lv}.npy'], 'm', True)
    cA[n] = dA * nl(kterm(QA, BA, [Rs])[:, 0]); cBA[n] = {K: dB * nl(v) for K, v in cond(QA, QB, BA, BB, KS).items()}
    cM[n] = {R: dM * nl(v) for R, v in zip(RM, kterm(QM, BM, RM).T)}
pz = {n: PZ[f'pose__{n}'] for n in pick}
selK = {K: float(np.mean([auroc(cA[idn] + cBA[idn][K] + pz[idn], cA[s] + cBA[s][K] + pz[s]) for s in sel])) for K in KS}
selR = {R: float(np.mean([auroc(cM[idn][R], cM[s][R]) for s in sel])) for R in RM}
Kc = max(KS, key=lambda K: (selK[K], -K)); Rm = max(RM, key=lambda R: (selR[R], -R))
choice = {'tag': tag, 'level': lv, 'R_star': Rs, 'R_upright': RU, 'K': Kc, 'R_c': max(1, min(RU, Kc // 5)), 'R_m': Rm, 'sel_K': selK, 'sel_Rm': selR,
          'd_sel_seed0': {'A': dA, 'B': dB, 'M': dM}}
print(tag, 'choice', {k: v for k, v in choice.items() if k not in ('sel_K', 'sel_Rm')}, f'{time.time() - t0:.0f}s', flush=True)
print('   sel K', {k: round(v, 2) for k, v in selK.items()}, ' sel R_m', {k: round(v, 2) for k, v in selR.items()}, flush=True)
del BA, BB, BM; torch.cuda.empty_cache()
os.makedirs(f'{ROOT}/runs/vl/ctrl', exist_ok=True)
json.dump(choice, open(f'{ROOT}/runs/vl/ctrl/{tag}_choice.json', 'w'), indent=1)

# ---- development split, noise seeds 0-2 (B: runs/ok features of Upright kNN; A and the channel means: runs/vl/featf at sigma_0)
for seed in (0, 1, 2):
    OD = f'{ROOT}/runs/ok/s{seed}/{tag}/k0'; FF = f'{ROOT}/runs/vl/featf/s{seed}/{tag}/k0'
    U = np.load(f'{ROOT}/runs/kx/export/{tag}_s{seed}_fresh.npz')
    assert (np.load(f'{OD}/bank__idx.npy') == np.load(f'{FF}/bank__idx.npy')).all(), 'bank order'
    BA = space([f'{FF}/bank__L{lv}.npy'], 'g', False); BB = space([f'{OD}/bank__L{l}.npy' for l in (0, 1, 2)], 'g', True); BM = space([f'{FF}/bank__L{lv}.npy'], 'm', False)
    dA, dB, dM = lid(BA), lid(BB), lid(BM); out = {'d': np.array([dA, dB, dM])}
    for n in FRESH:
        ui = U[f'idx__{n}']; assert (np.load(f'{FF}/{n}__idx.npy') == ui).all() and (np.load(f'{OD}/{n}__idx.npy') == ui).all(), n
        QA = space([f'{FF}/{n}__L{lv}.npy'], 'g', False); QB = space([f'{OD}/{n}__L{l}.npy' for l in (0, 1, 2)], 'g', True); QM = space([f'{FF}/{n}__L{lv}.npy'], 'm', False)
        out[f'cA__{n}'] = dA * nl(kterm(QA, BA, [Rs])[:, 0]); out[f'cBA__{n}'] = dB * nl(cond(QA, QB, BA, BB, [Kc])[Kc])
        out[f'cB__{n}'] = dB * nl(kterm(QB, BB, [RU])[:, 0]); out[f'cM__{n}'] = dM * nl(kterm(QM, BM, [Rm])[:, 0]); out[f'idx__{n}'] = ui
    np.savez(f'{ROOT}/runs/vl/ctrl/{tag}_s{seed}_fresh.npz', **out)
    del BA, BB, BM; torch.cuda.empty_cache()
    print(tag, 'seed', seed, 'd A/B/M', round(dA, 2), round(dB, 2), round(dM, 2), f'{time.time() - t0:.0f}s', flush=True)
print('VL_CTRL_DONE', tag, f'{time.time() - t0:.0f}s', flush=True)
