# e9_ana.py -- PROTOCOL_E9.md (written before the results were read): the frame term log q(R0 | x) of Upright kNN added to other
#   detectors on the final evaluation images, with the bootstrap of ours_vs_all.py (2,000 replicates, rng 20261031; win / loss when the
#   95% interval of the AUROC difference excludes 0 and |gap| >= 0.05).
#   Rows per ID dataset and noise seed:
#     CFS (kNN)            -rbar (stored)                 Upright CFS (kNN)    -d_CFS log rbar + pose      (d_CFS: e9_space.py)
#     kNN (basic)          c_10 (stored)                  Upright kNN (basic)  -d_L1 log sqrt(2(1-c_10)) + pose
#     CFS (diag)           stored                         Upright CFS (diag)   ll + pose                   (ll: e9_space.py)
#     z-sum fusions        z(m) + z(pose) for m in the eight statistics, CFS (diag), CFS (kNN), RotCls4 (z from fresh-split ID images)
#     Ours, kNN term, Orientation term for reference.
#   Comparisons: upright version vs its base (paired bootstrap), each fusion vs its base, and Ours vs every row.
#   usage: python e9_ana.py [<e9 results dir: runs/kx/e9_space_<tag>_s<seed>.npz>]   -> e9_ana.json
import os, sys, json, pickle
import numpy as np
SPLIT = 'final2'; os.environ['SPLIT'] = SPLIT
SCR = os.environ.get('UPKNN_ANALYSIS', os.path.expanduser('~/upknn/analysis'))
HERE = os.path.dirname(os.path.abspath(__file__)); RES = sys.argv[1] if len(sys.argv) > 1 else f'{SCR}/final/e9/res'
sys.path.insert(0, f'{SCR}/final/ana'); sys.path.insert(0, f'{SCR}/final/ana_ok')
from ana_common import *
from boot import counts, boot_auroc
import ok_common as OC
B = 2000; SEEDS = (0, 1, 2)
COMX = json.load(open(f'{SCR}/final/eu_stage/push_final2/final2_idx/final2_idx.json'))
COMF = json.load(open(f'{SCR}/final/eu_stage/push/fresh_idx/fresh_idx.json'))
EXP = f'{SCR}/final/data_kx/export'; CFSD = f'{SCR}/final/data_kx/cfs'
BL = {sp: {sd: pickle.load(open(f'{SCR}/final/ana/baselines_perimage_{sp}_s{sd}.pkl', 'rb')) for sd in SEEDS} for sp in ('final2', 'fresh')}
def nm(tag, s): return setname(tag, s)
def kx(tag, s, seed, key, split=SPLIT):
    Z = np.load(f'{EXP}/{tag}_s{seed}_{split}.npz'); n = f'{split}-{nm(tag, s)}'; com = COMX if split == 'final2' else COMF
    return sub(Z[f'{key}__{n}'].astype(np.float64), Z[f'idx__{n}'], com[tag][s])
def cfs(tag, s, seed, head, split=SPLIT):
    z = np.load(f'{CFSD}/s{seed}/{tag}/{split}-{nm(tag, s)}__scores.npz'); com = COMX if split == 'final2' else COMF
    return sub(z[head].astype(np.float64), z['idx'], com[tag][s])
def ctknn(tag, s, seed, split=SPLIT):
    os.environ['SPLIT'] = split; D = OC.load(tag, f'{split}-{nm(tag, s)}', seed); v = OC.comp_value(D, ('C1', 'g', 'L1', 0, 10)); os.environ['SPLIT'] = SPLIT
    com = COMX if split == 'final2' else COMF; return sub(np.asarray(v, np.float64), D[0]['idx'], com[tag][s])
def rc(tag, s, split=SPLIT):
    z = np.load(f'{DATA}/runs/rotcls/{tag}/scores_{split}/{nm(tag, s)}.npz'); com = COMX if split == 'final2' else COMF
    return sub(z['rot4'].astype(np.float64), z['idx'], com[tag][s])
def e9(tag, seed): return np.load(f'{RES}/runs/kx/e9_space_{tag}_s{seed}.npz')
def ll_cfs(tag, s, seed):
    z = e9(tag, seed); n = f'{SPLIT}-{nm(tag, s)}'; return sub(z[f'll__{n}'].astype(np.float64), z[f'idx__{n}'], COMX[tag][s])
class Norm:
    def __init__(s, ref): s.mu, s.sd = ref.mean(), ref.std() + 1e-12
    def z(s, v): return (v - s.mu) / s.sd
def verdict(lo, hi, gap):
    if abs(gap) < 0.05: return 'tie'
    return 'win' if lo > 0 else ('loss' if hi < 0 else 'tie')


def rows_for(tag, seed):
    """{row: function(set) -> ID-high scores on the common final2 images}; (base, upright) pairs listed in PAIRS"""
    z = e9(tag, seed); d_cfs, d_l1 = float(z['d_cfs']), float(z['d_l1'])
    pose = lambda s: kx(tag, s, seed, 'pose'); Np = Norm(kx(tag, '__id__', seed, 'pose', 'fresh'))
    rbar = lambda s: -cfs(tag, s, seed, 'knn')                                  # stored kNN score is -rbar
    r_l1 = lambda s: np.sqrt(2 * (1 - np.minimum(ctknn(tag, s, seed), 1 - 1e-9)))
    R = {'Ours': lambda s: kx(tag, s, seed, 'score'), 'kNN term': lambda s: kx(tag, s, seed, 'content'), 'Orientation term': pose,
         'CFS (kNN)': lambda s: -rbar(s), 'Upright CFS (kNN)': lambda s: -d_cfs * np.log(rbar(s)) + pose(s),
         'kNN (basic)': lambda s: ctknn(tag, s, seed), 'Upright kNN (basic)': lambda s: -d_l1 * np.log(r_l1(s)) + pose(s),
         'CFS (diag)': lambda s: cfs(tag, s, seed, 'diag'), 'CFS (diag, Gaussian)': lambda s: ll_cfs(tag, s, seed), 'Upright CFS (diag)': lambda s: ll_cfs(tag, s, seed) + pose(s)}
    # CFS (diag, Gaussian): the Gaussian log-density -1/2 sum_j (z_j - mu_j)^2 / v_j over all descriptor dimensions, the base of Upright CFS (diag);
    #   it ranks as the official slot-mean score only when the two slots have the same width (on SVHN the slots have 1,024 and 256 dimensions)
    # z-sum fusions with the frame term (normalisation from the ID images of the development split)
    fus = {}
    for m in BASELINES:
        if m in BL['final2'][seed][tag]['__id__'] and m in BL['fresh'][seed][tag]['__id__']:
            Nm = Norm(BL['fresh'][seed][tag]['__id__'][m]); b = (lambda s, m=m: BL['final2'][seed][tag]['__id__' if s == '__id__' else s][m])
            R[m] = b; R[f'{m} + frame (z)'] = (lambda s, b=b, Nm=Nm: Nm.z(b(s)) + Np.z(pose(s))); fus[m] = f'{m} + frame (z)'
    for m, f, ff in (('CFS (diag)', lambda s: cfs(tag, s, seed, 'diag'), lambda s: cfs(tag, s, seed, 'diag', 'fresh')),
                     ('CFS (kNN)', lambda s: cfs(tag, s, seed, 'knn'), lambda s: cfs(tag, s, seed, 'knn', 'fresh')),
                     ('kNN (basic)', lambda s: ctknn(tag, s, seed), lambda s: ctknn(tag, s, seed, 'fresh'))):
        Nm = Norm(ff('__id__')); R[f'{m} + frame (z)'] = (lambda s, f=f, Nm=Nm: Nm.z(f(s)) + Np.z(pose(s))); fus[m] = f'{m} + frame (z)'
    if os.path.exists(f'{DATA}/runs/rotcls/{tag}/scores_{SPLIT}/{nm(tag, "__id__")}.npz') and seed == 0:
        Nr = Norm(rc(tag, '__id__', 'fresh')); R['RotCls4'] = lambda s: rc(tag, s); R['RotCls4 + frame (z)'] = lambda s: Nr.z(rc(tag, s)) + Np.z(pose(s)); fus['RotCls4'] = 'RotCls4 + frame (z)'
    return R, dict(d_cfs=d_cfs, d_l1=d_l1, D_cfs=int(z['D_cfs']), N_bank=int(z['N_bank'])), fus


PAIRS = [('CFS (kNN)', 'Upright CFS (kNN)'), ('kNN (basic)', 'Upright kNN (basic)'), ('CFS (diag, Gaussian)', 'Upright CFS (diag)'), ('CFS (diag)', 'Upright CFS (diag)'), ('kNN term', 'Ours')]
rng = np.random.default_rng(20261031)
OUT = {'split': SPLIT, 'B': B, 'pairs': {}, 'ids': {}, 'vflip': {}, 'space': {}, 'fusions': {}}
for tag, T in TIERS.items():
    RS = {}; SP = {}; FUS = None
    for sd in SEEDS:
        try: RS[sd], SP[sd], fus = rows_for(tag, sd); FUS = FUS or fus
        except FileNotFoundError as e: print(tag, 'seed', sd, 'missing', e); continue
    if not RS: continue
    seeds = sorted(RS); names = list(RS[seeds[0]]); OUT['space'][tag] = {sd: SP[sd] for sd in seeds}; OUT['fusions'][tag] = FUS
    allpairs = PAIRS + [(b, u) for b, u in FUS.items()] + [(r, 'Ours') for r in names if r != 'Ours']
    ri = {r: {sd: RS[sd][r]('__id__') for sd in seeds if r in RS[sd]} for r in names}
    Wa = counts(rng, len(ri['Ours'][seeds[0]]), B); pr = {}
    for o in T['ood']:
        ro = {r: {sd: RS[sd][r](o) for sd in seeds if r in RS[sd]} for r in names}; Wb = counts(rng, len(ro['Ours'][seeds[0]]), B)
        pts = {r: [100 * auroc(ri[r][sd], ro[r][sd]) for sd in ri[r]] for r in names}
        bb = {r: np.mean([boot_auroc(ri[r][sd], ro[r][sd], Wa, Wb) for sd in ri[r]], 0) for r in names}
        pr[o] = {r: dict(auroc=float(np.mean(pts[r])), seeds=pts[r]) for r in names}; pr[o]['_vs'] = {}
        for base, up in allpairs:
            if base not in names or up not in names: continue
            d = bb[up] - bb[base]; lo, hi = np.percentile(d, [2.5, 97.5]); gap = pr[o][up]['auroc'] - pr[o][base]['auroc']
            pr[o]['_vs'][f'{up} vs {base}'] = dict(diff=gap, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap))
    OUT['pairs'][tag] = pr; nat = NAT[tag]
    OUT['ids'][tag] = {r: float(np.mean([pr[o][r]['auroc'] for o in nat])) for r in names}
    vf = [o for o in T['ood'] if 'vflip' in o]; OUT['vflip'][tag] = {r: float(np.mean([pr[o][r]['auroc'] for o in vf])) for r in names}
    print(f"{ID_NAME[tag]:13s} d_cfs {SP[seeds[0]]['d_cfs']:.1f} d_l1 {SP[seeds[0]]['d_l1']:.1f} | " + ' '.join(f"{r}: {OUT['ids'][tag][r]:.1f}" for r in ('CFS (kNN)', 'Upright CFS (kNN)', 'kNN (basic)', 'Upright kNN (basic)', 'CFS (diag)', 'CFS (diag, Gaussian)', 'Upright CFS (diag)', 'kNN term', 'Ours')), flush=True)
rows_all = sorted({r for t in OUT['ids'] for r in OUT['ids'][t]})
OUT['mean'] = {r: float(np.mean([OUT['ids'][t][r] for t in OUT['ids'] if r in OUT['ids'][t]])) for r in rows_all}
OUT['mean_tiers'] = {r: [t for t in OUT['ids'] if r in OUT['ids'][t]] for r in rows_all}
OUT['vflip_mean'] = {r: float(np.mean([OUT['vflip'][t][r] for t in OUT['vflip'] if r in OUT['vflip'][t]])) for r in rows_all}
allp = [(t, o) for t in OUT['pairs'] for o in OUT['pairs'][t] if o in NAT[t]]
keys = sorted({k for t, o in allp for k in OUT['pairs'][t][o]['_vs']})
OUT['wtl_nat'] = {k: {v: sum(OUT['pairs'][t][o]['_vs'][k]['verdict'] == v for t, o in allp if k in OUT['pairs'][t][o]['_vs']) for v in ('win', 'tie', 'loss')} for k in keys}
OUT['big_nat'] = {k: {'win1': sum(OUT['pairs'][t][o]['_vs'][k]['verdict'] == 'win' and OUT['pairs'][t][o]['_vs'][k]['diff'] >= 1 for t, o in allp if k in OUT['pairs'][t][o]['_vs']),
                      'loss1': sum(OUT['pairs'][t][o]['_vs'][k]['verdict'] == 'loss' and OUT['pairs'][t][o]['_vs'][k]['diff'] <= -1 for t, o in allp if k in OUT['pairs'][t][o]['_vs'])} for k in keys}
print('== mean over the ID datasets (natural OOD sets) and flip:')
for r in rows_all: print(f"  {r:28s} {OUT['mean'][r]:5.1f}  flip {OUT['vflip_mean'][r]:5.1f}  ({len(OUT['mean_tiers'][r])} ID datasets)")
print('== upright / fusion vs base, W/T/L over the natural pairs (wins / losses of >= 1 point):')
for k in keys:
    w = OUT['wtl_nat'][k]; b = OUT['big_nat'][k]; print(f"  {k:48s} {w['win']}/{w['tie']}/{w['loss']}  ({b['win1']}/{b['loss1']})")
json.dump(OUT, open(f'{HERE}/e9_ana.json', 'w'), indent=1)
print('written', f'{HERE}/e9_ana.json')
