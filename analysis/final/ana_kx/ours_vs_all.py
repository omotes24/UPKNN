# ours_vs_all.py -- the orientation-completed kNN ("Ours", kx_export.py, noise seeds 0-2) against every comparison row on one split,
#   with the bootstrap protocol of permethod.py (2000 replicates, rng 20261031, ID images resampled once per ID, OOD per set, in TIERS
#   order; per pair CI of AUROC(Ours) - AUROC(row); win/loss if the CI excludes 0 and |gap| >= 0.05).
#   Rows: the eight comparison methods (per-image caches of permethod.py), CFS(1x2) with its diagonal and kNN heads (3 seeds; added after
#   freezing, like the RotCls rows), Euphonium (3 seeds), the PROTOCOL_CT kNN (3 seeds), the
#   tuned kNN (per-ID choice of ok_prof_sel.json, 3 seeds), RotCls4/RotCls2/Lambda90 (1 seed), and the content / pose terms of Ours.
#   usage: SPLIT=fresh python ours_vs_all.py      (SPLIT=final2 only after the protocol is frozen)
import os, sys, json, pickle
import numpy as np
SPLIT = os.environ.get('SPLIT', 'fresh'); assert SPLIT in ('fresh', 'final2')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../ana'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../ana_ok')
os.environ['SPLIT'] = SPLIT
from ana_common import *
from boot import counts, boot_auroc
import ok_common as OC
B = 2000; SEEDS = (0, 1, 2)
COMX = json.load(open(f'{SCR}/final/eu_stage/push/fresh_idx/fresh_idx.json')) if SPLIT == 'fresh' else json.load(open(f'{SCR}/final/eu_stage/push_final2/final2_idx/final2_idx.json'))
EXP = f'{SCR}/final/data_kx/export'
SEL = json.load(open(f'{SCR}/final/ana_ok/ok_prof_sel.json')); PERID = {t: max(SEL, key=lambda k: SEL[k][t]) for t in TIERS}
PD = f'{SCR}/final/data_ok/prof'; SD = f'{SCR}/final/data_ok/sims'
def verdict(lo, hi, gap):
    if abs(gap) < 0.05: return 'tie'
    return 'win' if lo > 0 else ('loss' if hi < 0 else 'tie')
def nm(tag, s): return setname(tag, s)
def ours(tag, s, seed, key='score'):
    Z = np.load(f'{EXP}/{tag}_s{seed}_{SPLIT}.npz'); n = f'{SPLIT}-{nm(tag, s)}'
    return sub(Z[f'{key}__{n}'].astype(np.float64), Z[f'idx__{n}'], COMX[tag][s])
def tuned(tag, s, seed):
    n = f'{SPLIT}-{nm(tag, s)}'; Z = np.load(f'{PD}/s{seed}/{tag}__{n}.npz'); st, g, k, r = PERID[tag].split('_')
    v = Z[f'{st}_{g}_{k}'][:, list(Z['R']).index(int(r[1:]))].astype(np.float64)
    idx = np.load(f'{SD}/s{seed}/{tag}/k0/{n}.npz')['idx']; return sub(v, idx, COMX[tag][s])
def ctknn(tag, s, seed):
    D = OC.load(tag, f'{SPLIT}-{nm(tag, s)}', seed); v = OC.comp_value(D, ('C1', 'g', 'L1', 0, 10)); return sub(np.asarray(v, np.float64), D[0]['idx'], COMX[tag][s])
def eu(tag, s, seed):
    z = np.load(f'{DATA}/runs/eu/s{seed}/{tag}_{SPLIT}/{nm(tag, s)}.npz'); return sub(z['eu'], z['idx'], COMX[tag][s])
BL = {sd: pickle.load(open(f'{SCR}/final/ana/baselines_perimage_{SPLIT}_s{sd}.pkl', 'rb')) for sd in SEEDS}
CFSD = f'{SCR}/final/data_kx/cfs'                                    # CFS(1x2) of code/cfs_extract.py + cfs_score.py (added 2026-11-01, after freezing)
def cfs(tag, s, seed, head):
    z = np.load(f'{CFSD}/s{seed}/{tag}/{SPLIT}-{nm(tag, s)}__scores.npz'); return sub(z[head].astype(np.float64), z['idx'], COMX[tag][s])
ROWS3 = {'kNN tuned': tuned, 'kNN (CT)': ctknn, 'Euphonium': eu, 'Ours content': lambda t, s, sd: ours(t, s, sd, 'content'), 'Ours pose': lambda t, s, sd: ours(t, s, sd, 'pose'),
         'CFS': lambda t, s, sd: cfs(t, s, sd, 'diag'), 'CFS-kNN': lambda t, s, sd: cfs(t, s, sd, 'knn')}
rng = np.random.default_rng(20261031)
OUT = {'split': SPLIT, 'B': B, 'pairs': {}, 'ids': {}, 'rows': {}}
for tag, T in TIERS.items():
    oi = {sd: ours(tag, '__id__', sd) for sd in SEEDS}; Wa = counts(rng, len(oi[0]), B)
    ri = {r: {sd: f(tag, '__id__', sd) for sd in SEEDS} for r, f in ROWS3.items()}
    pr, bo, bm = {}, {}, {}
    for o in T['ood']:
        oo = {sd: ours(tag, o, sd) for sd in SEEDS}; Wb = counts(rng, len(oo[0]), B)
        opt = [100 * auroc(oi[sd], oo[sd]) for sd in SEEDS]; ob = np.mean([boot_auroc(oi[sd], oo[sd], Wa, Wb) for sd in SEEDS], 0); bo[o] = ob
        rows = {}
        def add(r, a_list, b_list):
            pts = [100 * auroc(a, b) for a, b in zip(a_list, b_list)]; bb = np.mean([boot_auroc(a, b, Wa, Wb) for a, b in zip(a_list, b_list)], 0)
            d = ob - bb; lo, hi = np.percentile(d, [2.5, 97.5]); gap = float(np.mean(opt) - np.mean(pts))
            rows[r] = dict(auroc=float(np.mean(pts)), seeds=pts, diff=gap, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap)); bm[(o, r)] = bb
        for r, f in ROWS3.items(): add(r, [ri[r][sd] for sd in SEEDS], [f(tag, o, sd) for sd in SEEDS])
        for m in BASELINES:
            sm = [sd for sd in SEEDS if BL[sd].get(tag) and m in BL[sd][tag].get(o, {}) and m in BL[sd][tag]['__id__']]
            if sm: add(m, [BL[sd][tag]['__id__'][m] for sd in sm], [BL[sd][tag][o][m] for sd in sm])
        rc = f'{DATA}/runs/rotcls/{tag}/scores_{SPLIT}'                 # learned rotation classifiers (ResNet-18 from scratch, one seed)
        fi, fo = f'{rc}/{nm(tag, "__id__")}.npz', f'{rc}/{nm(tag, o)}.npz'
        if os.path.exists(fi) and os.path.exists(fo):
            zi, zo = np.load(fi), np.load(fo)
            for r, key in (('RotCls4', lambda z: z['rot4']), ('RotCls2', lambda z: z['bin90'] + z['bin180'])):
                add(r, [sub(key(zi).astype(np.float64), zi['idx'], COMX[tag]['__id__'])], [sub(key(zo).astype(np.float64), zo['idx'], COMX[tag][o])])
        pr[o] = dict(ours=float(np.mean(opt)), ours_seeds=opt, rows=rows)
    OUT['pairs'][tag] = pr
    nat = NAT[tag]; om = float(np.mean([pr[o]['ours'] for o in nat])); OUT['ids'][tag] = {'ours': om, 'rows': {}}
    for r in list(ROWS3) + BASELINES + ['RotCls4', 'RotCls2']:
        if not all(r in pr[o]['rows'] for o in nat): continue
        rm = float(np.mean([pr[o]['rows'][r]['auroc'] for o in nat])); dn = np.mean([bo[o] for o in nat], 0) - np.mean([bm[(o, r)] for o in nat], 0)
        lo, hi = np.percentile(dn, [2.5, 97.5]); OUT['ids'][tag]['rows'][r] = dict(mean=rm, diff=om - rm, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, om - rm))
    print(f"{ID_NAME[tag]:13s} Ours {om:5.1f}  " + ' '.join(f"{r[:9]} {OUT['ids'][tag]['rows'][r]['mean']:5.1f}" for r in OUT['ids'][tag]['rows']), flush=True)
allp = [(t, o, p) for t in OUT['pairs'] for o, p in OUT['pairs'][t].items()]
for r in list(ROWS3) + BASELINES + ['RotCls4', 'RotCls2']:
    vs = [p['rows'][r]['verdict'] for _, _, p in allp if r in p['rows']]
    if not vs: continue
    OUT['rows'][r] = dict(wins=vs.count('win'), ties=vs.count('tie'), losses=vs.count('loss'), pairs=len(vs),
                          mean_nat=float(np.mean([OUT['ids'][t]['rows'][r]['mean'] for t in TIERS if r in OUT['ids'][t]['rows']])))
OUT['mean_ours'] = float(np.mean([OUT['ids'][t]['ours'] for t in TIERS]))
print('mean Ours', round(OUT['mean_ours'], 2)); print({r: (v['wins'], v['ties'], v['losses'], round(v['mean_nat'], 2)) for r, v in OUT['rows'].items()})
json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'ours_vs_all_{SPLIT}.json'), 'w'), indent=1)
