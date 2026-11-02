# permethod.py -- Euphonium against each comparison method separately (the comparison of the ICML version): per pair the AUROC of
#   every method as the mean over its noise seeds (Euphonium: rules of seeds 0, 1, 2; GEPC, EigenScore: baselines.py --seed 0, 1, 2;
#   NLL / IC / TT: the PF-ODE probe seeds where the runs exist; MSMA type and DiffPath are deterministic), the 95% bootstrap confidence
#   interval of AUROC(Euphonium) - AUROC(method) (2000 replicates of the images, shared across methods and seeds; the ID images are
#   resampled once per replicate for all pairs of an ID), wins / ties / losses per method (CI excludes zero and the point gap does not
#   round to 0.0), the mean rank over the pairs, and, for reference, the same count against the best method of each pair (oracle).
#   SPLIT=fresh evaluates the held-out images (runs/eu/s*/<tag>_fresh, inputs tree final/inputs_fresh, images fresh_idx.json).
#   Writes permethod.json (permethod_fresh.json) and caches the per-image scores of the comparison methods per seed.
import os, sys, json, pickle, shutil
import numpy as np
from ana_common import *
from common import NLLRUN
import common, baselines_eval as BE
from boot import counts, boot_auroc

SPLIT = os.environ.get('SPLIT', '')
B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SEEDS = (0, 1, 2)
SRC = os.environ['EUPHONIUM_INPUTS']
SUF = f'_{SPLIT}' if SPLIT else ''
FRESH_INPUTS = f'{SCR}/final/inputs_{SPLIT or "fresh"}'
TMP = f'{SCR}/final/tmp_inputs_pm'
COMX = (json.load(open(f'{SCR}/final/eu_stage/push/fresh_idx/fresh_idx.json')) if SPLIT == 'fresh' else
        json.load(open(f'{SCR}/final/eu_stage/push_final2/final2_idx/final2_idx.json')) if SPLIT == 'final2' else COM)
OUTF = f'permethod{SUF}.json'


def verdict(lo, hi, gap):
    if abs(gap) < 0.05: return 'tie'
    return 'win' if lo > 0 else ('loss' if hi < 0 else 'tie')


def eu_scores(tag, name, seed):
    p = f'{DATA}/runs/eu/s{seed}/{tag}{SUF}/{name}.npz'
    z = np.load(p); return z['eu'], z['idx']


def onc(tag, s, score, idx):
    return sub(np.asarray(score), np.asarray(idx), COMX[tag][s])


def baseline_tree(seed):
    """inputs tree of one seed of the comparison methods: {tag: set of methods available at this seed}; the tree is built in TMP
    (symlinks), missing runs fall back to seed 0 so that baselines_eval.py runs, and their methods are left out of the seed"""
    if os.path.exists(TMP): shutil.rmtree(TMP)
    os.makedirs(TMP)
    base_src = FRESH_INPUTS if SPLIT else SRC
    for d in ('base', 'complexity', 'feat'):
        os.symlink(os.path.realpath(f'{SRC}/{d}' if d in ('complexity', 'feat') else f'{base_src}/{d}'), f'{TMP}/{d}')
    os.makedirs(f'{TMP}/base2'); os.makedirs(f'{TMP}/nll')
    valid = {}
    for tag in TIERS:
        v = {'MSMA*', 'DiffPath'} if seed == 0 else set()
        nsets = len(COMX[tag])
        for raw, m in (('gepc', 'GEPC'), ('eigen', 'EigenScore')):
            if seed == 0: src = f'{base_src}/base2/{raw}_{tag}'
            elif SPLIT: src = f'{FRESH_INPUTS}/base2/{raw}_{tag}_s{seed}'
            else: src = f'{DATA}/runs/base2/{raw}_{tag}_s{seed}'
            ok = os.path.isdir(src) and len([f for f in os.listdir(src) if '__cal' not in f]) >= nsets - 1      # vflip has no own file
            if ok: v.add(m)
            else: src = f'{base_src}/base2/{raw}_{tag}'
            os.symlink(os.path.realpath(src), f'{TMP}/base2/{raw}_{tag}')
        os.symlink(os.path.realpath(f'{base_src}/base2/diffpath_{tag}'), f'{TMP}/base2/diffpath_{tag}')
        if seed == 0: src = f'{base_src}/nll/F_{tag}'
        elif SPLIT: src = f'{FRESH_INPUTS}/nll/F_{tag}_s{seed}'
        else: src = f'{DATA}/runs/nll/F_{tag}_s{seed}'
        nreal = len([o for o in COMX[tag] if 'synth' not in o]) - 1
        if os.path.isdir(src) and len([f for f in os.listdir(src) if '__cal' not in f]) >= nreal: v |= {'NLL', 'IC-png', 'IC-j2k', 'TT'}
        else: src = f'{base_src}/nll/F_{tag}'
        os.symlink(os.path.realpath(src), f'{TMP}/nll/F_{tag}')
        valid[tag] = v
    common.INPUTS = TMP; BE.INPUTS = TMP
    return valid


def baseline_perimage(seed):
    """{tag: {'__id__' | set: {method: scores on the common images}}} of one seed (cached)"""
    cache = f'baselines_perimage{SUF}_s{seed}.pkl'
    if os.path.exists(cache): return pickle.load(open(cache, 'rb'))
    valid = baseline_tree(seed)
    out = {}
    for tag, T in TIERS.items():
        if not valid[tag]: out[tag] = {}; continue
        S, _ = BE.tier_scores(tag, T)
        out[tag] = {}
        for s in ['__id__'] + T['ood']:
            name = setname(tag, s)
            out[tag][s] = {m: sub(*S[name][m], COMX[tag][s]) for m in valid[tag] if m in S[name]}
    pickle.dump(out, open(cache, 'wb'))
    return out


BL = {s: baseline_perimage(s) for s in SEEDS}


# ---- further comparison rows: the exact ratio (PF-ODE), the learned rotation classifiers and the feature kNN (one seed each)
def eudiff_scores(tag, s, seed):
    """the error-difference form of Eq. (3): sum over the atoms of e_k(T x) - e_k(x) (same runs, same seed), on the common images"""
    p = f'{DATA}/runs/eu/s{seed}/{tag}{SUF}/{setname(tag, s)}.npz'
    z = np.load(p, allow_pickle=True); ks = [int(str(a).split(':')[1]) for a in z['atoms']]; scales = [int(k) for k in z['scales']]
    d = sum(z['e_rot'][:, j].astype(np.float64) - z['e_up'][:, scales.index(k)].astype(np.float64) for j, k in enumerate(ks))
    return sub(d, z['idx'], COMX[tag][s])


def extra_scores(tag, s):
    """{row: scores on the common images} of set s ('__id__' or an OOD set) of tier tag"""
    T = TIERS[tag]; name = setname(tag, s); out = {}
    nr = f'{DATA}/runs/nll/F_{tag}_fresh' if SPLIT == 'fresh' else (None if SPLIT else (f'{SRC}/nll/{NLLRUN[tag]}' if tag in NLLRUN else None))
    if nr and os.path.exists(f'{nr}/{name}.npz') and os.path.exists(f'{nr}_rot90/{name}.npz'):
        N0, R0 = np.load(f'{nr}/{name}.npz'), np.load(f'{nr}_rot90/{name}.npz')
        if np.array_equal(N0['idx'], R0['idx']) and set(COMX[tag][s]) <= set(N0['idx'].tolist()):
            out['Lambda90'] = sub(N0['ll'].astype(np.float64) - R0['ll'].astype(np.float64), N0['idx'], COMX[tag][s])
    rc = f'{DATA}/runs/rotcls/{tag}/{"scores_" + SPLIT if SPLIT else "scores"}/{name}.npz'
    if os.path.exists(rc):
        z = np.load(rc)
        if set(COMX[tag][s]) <= set(z['idx'].tolist()):
            out['RotCls4'] = sub(z['rot4'].astype(np.float64), z['idx'], COMX[tag][s])
            out['RotCls2'] = sub(z['bin90'].astype(np.float64) + z['bin180'].astype(np.float64), z['idx'], COMX[tag][s])
    if SPLIT == 'fresh':
        kf = f'{DATA}/runs/ct_fresh/{tag}/k0/fresh-{name}.npz'
        if os.path.exists(kf):
            z = np.load(kf); K = z['K'][:, 1, 0, 1, 1].astype(np.float64)           # group L1, +eps, bank 50k, 10th neighbour
            if set(COMX[tag][s]) <= set(z['idx'].tolist()): out['kNN'] = sub(K, z['idx'], COMX[tag][s])
    else:
        pf = f'{SCR}/ct/data/packfin_{tag}.npz'
        if os.path.exists(pf):
            z = np.load(pf); key = 'fin-' + name
            if 'K__' + key in z.files:
                K = z['K__' + key][:, 0, 1, 0, 1, 1].astype(np.float64)
                if set(COMX[tag][s]) <= set(z['idx__' + key].tolist()): out['kNN'] = sub(K, z['idx__' + key], COMX[tag][s])
    return out


EXTRA = ['Lambda90', 'RotCls4', 'RotCls2', 'kNN']
EXTRA_ALL = EXTRA + ['EuDiff']
rng = np.random.default_rng(20261031)
OUT = {'B': B, 'seeds': SEEDS, 'split': SPLIT or 'main', 'pairs': {}, 'methods': {}, 'ids': {}}
pair_boot = {}
for tag, T in TIERS.items():
    ida = {s: onc(tag, '__id__', *eu_scores(tag, T['id'], s)) for s in SEEDS}
    n_id = len(ida[0]); Wa = counts(rng, n_id, B)
    pr, bootE, bootM = {}, {}, {}
    for o in T['ood']:
        ob = {s: onc(tag, o, *eu_scores(tag, o, s)) for s in SEEDS}
        Wb = counts(rng, len(ob[0]), B)
        eu_pt = [100 * auroc(ida[s], ob[s]) for s in SEEDS]
        be = np.mean([boot_auroc(ida[s], ob[s], Wa, Wb) for s in SEEDS], 0)
        meth = {}
        for m in BASELINES:
            seeds_m = [s for s in SEEDS if BL[s].get(tag) and m in BL[s][tag].get(o, {}) and m in BL[s][tag]['__id__']]
            if not seeds_m: continue
            pts = [100 * auroc(BL[s][tag]['__id__'][m], BL[s][tag][o][m]) for s in seeds_m]
            bm = np.mean([boot_auroc(BL[s][tag]['__id__'][m], BL[s][tag][o][m], Wa, Wb) for s in seeds_m], 0)
            d = be - bm; lo, hi = np.percentile(d, [2.5, 97.5]); gap = float(np.mean(eu_pt) - np.mean(pts))
            meth[m] = dict(auroc=float(np.mean(pts)), seeds=pts, n_seeds=len(seeds_m), diff=gap, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap))
            bootM[(o, m)] = bm
        xi, xo = extra_scores(tag, '__id__'), extra_scores(tag, o)
        # the error-difference form (three seeds, like Euphonium)
        edi = {sd: eudiff_scores(tag, '__id__', sd) for sd in SEEDS}; edo = {sd: eudiff_scores(tag, o, sd) for sd in SEEDS}
        pts = [100 * auroc(edi[sd], edo[sd]) for sd in SEEDS]; bm = np.mean([boot_auroc(edi[sd], edo[sd], Wa, Wb) for sd in SEEDS], 0)
        d = be - bm; lo, hi = np.percentile(d, [2.5, 97.5]); gap = float(np.mean(eu_pt) - np.mean(pts))
        meth['EuDiff'] = dict(auroc=float(np.mean(pts)), seeds=pts, n_seeds=len(SEEDS), diff=gap, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap))
        bootM[(o, 'EuDiff')] = bm
        for m in EXTRA:
            if m not in xi or m not in xo: continue
            pt = 100 * auroc(xi[m], xo[m]); bm = boot_auroc(xi[m], xo[m], Wa, Wb)
            d = be - bm; lo, hi = np.percentile(d, [2.5, 97.5]); gap = float(np.mean(eu_pt) - pt)
            meth[m] = dict(auroc=float(pt), seeds=[float(pt)], n_seeds=1, diff=gap, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap))
            bootM[(o, m)] = bm
        best = max((m for m in meth if m in BASELINES), key=lambda m: meth[m]['auroc'])
        d = be - bootM[(o, best)]; lo, hi = np.percentile(d, [2.5, 97.5]); gap = float(np.mean(eu_pt) - meth[best]['auroc'])
        scores = {'Euphonium': float(np.mean(eu_pt)), **{m: v['auroc'] for m, v in meth.items() if m in BASELINES}}
        order = sorted(scores, key=lambda k: -scores[k])
        ranks = {}
        for k in scores:                                     # average rank for exact ties
            same = [j for j, kk in enumerate(order) if abs(scores[kk] - scores[k]) < 1e-9]
            ranks[k] = float(np.mean(same) + 1)
        pr[o] = dict(eu=float(np.mean(eu_pt)), eu_sd=float(np.std(eu_pt, ddof=1)), eu_seeds=eu_pt, methods=meth, best=best,
                     oracle=dict(diff=gap, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap)), ranks=ranks)
        bootE[o] = be
    OUT['pairs'][tag] = pr
    nat = NAT[tag]
    OUT['ids'][tag] = dict(eu=float(np.mean([pr[o]['eu'] for o in nat])), eu_sd=float(np.std([np.mean([pr[o]['eu_seeds'][i] for o in nat]) for i in range(3)], ddof=1)),
                           methods={m: float(np.mean([pr[o]['methods'][m]['auroc'] for o in nat])) for m in BASELINES + EXTRA_ALL if all(m in pr[o]['methods'] for o in nat)})
    for m in BASELINES + EXTRA_ALL:
        if m not in OUT['ids'][tag]['methods']: continue
        dn = np.mean([bootE[o] for o in nat], 0) - np.mean([bootM[(o, m)] for o in nat], 0); lo, hi = np.percentile(dn, [2.5, 97.5])
        gap = OUT['ids'][tag]['eu'] - OUT['ids'][tag]['methods'][m]
        OUT['ids'][tag].setdefault('vs', {})[m] = dict(diff=float(gap), ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap))
    v = [pr[o]['oracle']['verdict'] for o in T['ood']]
    print(f"{ID_NAME[tag]:13s} Eu {OUT['ids'][tag]['eu']:5.1f}  oracle W/T/L {v.count('win')}/{v.count('tie')}/{v.count('loss')}  " +
          ' '.join(f"{m[:5]} {OUT['ids'][tag]['methods'].get(m, float('nan')):5.1f}" for m in BASELINES), flush=True)
allp = [(t, o, p) for t in OUT['pairs'] for o, p in OUT['pairs'][t].items()]
for m in BASELINES + EXTRA_ALL:
    vs = [p['methods'][m]['verdict'] for _, _, p in allp if m in p['methods']]
    if not vs: continue
    OUT['methods'][m] = dict(wins=vs.count('win'), ties=vs.count('tie'), losses=vs.count('loss'), pairs=len(vs),
                             mean_nat=float(np.mean([OUT['ids'][t]['methods'][m] for t in TIERS if m in OUT['ids'][t]['methods']])),
                             ids_with=len([t for t in TIERS if m in OUT['ids'][t]['methods']]),
                             mean_rank=float(np.mean([p['ranks'][m] for _, _, p in allp if m in p['ranks']])) if m in BASELINES else None,
                             n_seeds=sorted(set(p['methods'][m]['n_seeds'] for _, _, p in allp if m in p['methods'])))
ov = [p['oracle']['verdict'] for _, _, p in allp]
OUT['summary'] = dict(mean_eu=float(np.mean([OUT['ids'][t]['eu'] for t in TIERS])), eu_mean_rank=float(np.mean([p['ranks']['Euphonium'] for _, _, p in allp])),
                      oracle=dict(wins=ov.count('win'), ties=ov.count('tie'), losses=ov.count('loss')), pairs=len(allp),
                      min_wins=min(OUT['methods'][m]['wins'] for m in BASELINES), max_wins=max(OUT['methods'][m]['wins'] for m in BASELINES),
                      min_losses=min(OUT['methods'][m]['losses'] for m in BASELINES), max_losses=max(OUT['methods'][m]['losses'] for m in BASELINES),
                      strongest=max(BASELINES, key=lambda m: OUT['methods'][m]['mean_nat']))
print('per method W/T/L:', {m: (v['wins'], v['ties'], v['losses'], v['pairs'], round(v['mean_rank'], 2) if v['mean_rank'] else None, round(v['mean_nat'], 1), v['ids_with']) for m, v in OUT['methods'].items()})
print('summary:', json.dumps(OUT['summary']))
json.dump(OUT, open(OUTF, 'w'), indent=1)
