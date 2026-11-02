# fpr95_all.py -- FPR at 95% TPR (ID = positive, the threshold keeps 95% of the ID images) of every row of Table 1 and of Vertical Limit,
#   on the same images and seeds as ours_vs_all.py / vl_vs_uknn.py (per pair: mean over the available noise seeds; per ID: mean over the
#   natural OOD sets; 6-ID mean).  The AUROC means are recomputed alongside as a check against ours_vs_all_<split>.json.
#   usage: SPLIT=final2 python fpr95_all.py
import os, sys, json, pickle
import numpy as np
SPLIT = os.environ.get('SPLIT', 'final2'); assert SPLIT in ('fresh', 'final2')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../ana'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../ana_ok')
os.environ['SPLIT'] = SPLIT
from ana_common import *
import ok_common as OC
SEEDS = (0, 1, 2)
COMX = json.load(open(f'{SCR}/final/eu_stage/push/fresh_idx/fresh_idx.json')) if SPLIT == 'fresh' else json.load(open(f'{SCR}/final/eu_stage/push_final2/final2_idx/final2_idx.json'))
EXP = f'{SCR}/final/data_kx/export'; EXPV = f'{SCR}/final/data_vl/export'
SEL = json.load(open(f'{SCR}/final/ana_ok/ok_prof_sel.json')); PERID = {t: max(SEL, key=lambda k: SEL[k][t]) for t in TIERS}
PD = f'{SCR}/final/data_ok/prof'; SD = f'{SCR}/final/data_ok/sims'; CFSD = f'{SCR}/final/data_kx/cfs'
nm = setname


def exp(path, tag, s, seed, key='score'):
    Z = np.load(f'{path}/{tag}_s{seed}_{SPLIT}.npz'); n = f'{SPLIT}-{nm(tag, s)}'
    return sub(Z[f'{key}__{n}'].astype(np.float64), Z[f'idx__{n}'], COMX[tag][s])


def tuned(tag, s, seed):
    n = f'{SPLIT}-{nm(tag, s)}'; Z = np.load(f'{PD}/s{seed}/{tag}__{n}.npz'); st, g, k, r = PERID[tag].split('_')
    v = Z[f'{st}_{g}_{k}'][:, list(Z['R']).index(int(r[1:]))].astype(np.float64)
    idx = np.load(f'{SD}/s{seed}/{tag}/k0/{n}.npz')['idx']; return sub(v, idx, COMX[tag][s])


def ctknn(tag, s, seed):
    D = OC.load(tag, f'{SPLIT}-{nm(tag, s)}', seed); v = OC.comp_value(D, ('C1', 'g', 'L1', 0, 10)); return sub(np.asarray(v, np.float64), D[0]['idx'], COMX[tag][s])


def cfs(tag, s, seed, head):
    z = np.load(f'{CFSD}/s{seed}/{tag}/{SPLIT}-{nm(tag, s)}__scores.npz'); return sub(z[head].astype(np.float64), z['idx'], COMX[tag][s])


ROWS3 = {'Ours': lambda t, s, sd: exp(EXP, t, s, sd), 'VL': lambda t, s, sd: exp(EXPV, t, s, sd), 'kNN tuned': tuned, 'kNN (CT)': ctknn,
         'CFS': lambda t, s, sd: cfs(t, s, sd, 'diag'), 'CFS-kNN': lambda t, s, sd: cfs(t, s, sd, 'knn')}
BL = {sd: pickle.load(open(f'{SCR}/final/ana/baselines_perimage_{SPLIT}_s{sd}.pkl', 'rb')) for sd in SEEDS}
OUT = {'split': SPLIT, 'pairs': {}, 'ids': {}, 'mean': {}}
for tag, T in TIERS.items():
    pr = {}
    for o in T['ood']:
        pr[o] = {}
        for r, f in ROWS3.items():
            a = [f(tag, '__id__', sd) for sd in SEEDS]; b = [f(tag, o, sd) for sd in SEEDS]
            pr[o][r] = {'fpr95': float(np.mean([100 * fpr95(x, y) for x, y in zip(a, b)])), 'auroc': float(np.mean([100 * auroc(x, y) for x, y in zip(a, b)]))}
        for m in BASELINES:
            sm = [sd for sd in SEEDS if BL[sd].get(tag) and m in BL[sd][tag].get(o, {}) and m in BL[sd][tag]['__id__']]
            if sm: pr[o][m] = {'fpr95': float(np.mean([100 * fpr95(BL[sd][tag]['__id__'][m], BL[sd][tag][o][m]) for sd in sm])),
                               'auroc': float(np.mean([100 * auroc(BL[sd][tag]['__id__'][m], BL[sd][tag][o][m]) for sd in sm]))}
        rc = f'{DATA}/runs/rotcls/{tag}/scores_{SPLIT}'; fi, fo = f'{rc}/{nm(tag, "__id__")}.npz', f'{rc}/{nm(tag, o)}.npz'
        if os.path.exists(fi) and os.path.exists(fo):
            zi, zo = np.load(fi), np.load(fo)
            for r, key in (('RotCls4', lambda z: z['rot4']),):
                x = sub(key(zi).astype(np.float64), zi['idx'], COMX[tag]['__id__']); y = sub(key(zo).astype(np.float64), zo['idx'], COMX[tag][o])
                pr[o][r] = {'fpr95': 100 * fpr95(x, y), 'auroc': 100 * auroc(x, y)}
    OUT['pairs'][tag] = pr
    rows = sorted({r for o in NAT[tag] for r in pr[o]}, key=lambda r: list(pr[NAT[tag][0]]).index(r) if r in pr[NAT[tag][0]] else 99)
    OUT['ids'][tag] = {r: {'fpr95': float(np.mean([pr[o][r]['fpr95'] for o in NAT[tag]])), 'auroc': float(np.mean([pr[o][r]['auroc'] for o in NAT[tag]]))}
                       for r in rows if all(r in pr[o] for o in NAT[tag])}
    print(f"{ID_NAME[tag]:13s} " + ' '.join(f"{r[:8]} {v['fpr95']:5.1f}" for r, v in OUT['ids'][tag].items()), flush=True)
for r in OUT['ids']['edmc10']:
    if all(r in OUT['ids'][t] for t in TIERS):
        OUT['mean'][r] = {'fpr95': float(np.mean([OUT['ids'][t][r]['fpr95'] for t in TIERS])), 'auroc': float(np.mean([OUT['ids'][t][r]['auroc'] for t in TIERS]))}
print('6-ID mean FPR95 / AUROC:', {r: (round(v['fpr95'], 1), round(v['auroc'], 2)) for r, v in OUT['mean'].items()})
vf = {r: float(np.mean([OUT['pairs'][t][TIERS[t]['id'] + '__vflip'][r]['fpr95'] for t in TIERS if r in OUT['pairs'][t][TIERS[t]['id'] + '__vflip']])) for r in OUT['mean']}
OUT['vflip_fpr95'] = vf; print('upside-down FPR95 mean:', {r: round(v, 1) for r, v in vf.items()})
json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'fpr95_all_{SPLIT}.json'), 'w'), indent=1)
