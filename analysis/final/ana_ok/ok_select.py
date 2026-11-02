# ok_select.py -- PROTOCOL_OK.md selection (selection data only: devall-* ID development images vs the sel-* extra OOD sets), seed 0.
#   Two-fold cross-fitting of the ID development images (RandomState(0) permutation): calibrate the one-sided p-values on one half,
#   score the other half and every extra OOD set; AUROC averaged over the two folds.  Single-component candidates use the raw value
#   (T alone: minus the Mahalanobis distance fitted on the calibration half).  Writes ok_select.json.
import sys, json, hashlib, time
import numpy as np
from ok_common import *
NOIMNET = ('stl10_test', 'imagenette_val', 'in64_stl10_test')        # check: selection without the ImageNet-derived extra sets


def sel_sets(tag):
    rows = json.load(open(f'{SCR}/final/data_ok/sets_{tag}.json'))
    return [r[0] for r in rows if r[0].startswith('sel-')]


def tier_scores(tag, cands):
    t0 = time.time()
    Did = load(tag, 'devall-' + IDN[tag]); sets = sel_sets(tag); Ds = {s: load(tag, s) for s in sets}
    comps = sorted({x for c, _ in cands for x in c}, key=str)
    V = {c: (comp_value(Did, c), {s: comp_value(Ds[s], c) for s in sets}) for c in comps}
    n = len(Did[0]['idx']); perm = np.random.RandomState(0).permutation(n)
    folds = [(perm[:n // 2], perm[n // 2:]), (perm[n // 2:], perm[:n // 2])]
    LP = []
    for cal, ev in folds:
        lp = {}
        for c in comps:
            vi, vs = V[c]; xs = [vi[ev]] + [vs[s] for s in sets]
            lp[c] = calibrated_logp(c, vi[cal], xs)
        LP.append(lp)
    res = {}
    for comps_c, npass in cands:
        per = []
        for s_i, s in enumerate(sets):
            a = []
            for f, (cal, ev) in enumerate(folds):
                if len(comps_c) == 1 and comps_c[0][0] != 'T':
                    vi, vs = V[comps_c[0]]; a.append(auroc(vi[ev], vs[s]))
                else:
                    A = sum(LP[f][c][0] for c in comps_c); B = sum(LP[f][c][1 + s_i] for c in comps_c)
                    a.append(auroc(A, B))
            per.append(100 * np.mean(a))
        res[comps_c] = dict(per_set=dict(zip(sets, per)), mean=float(np.mean(per)),
                            mean_noimnet=float(np.mean([p for s, p in zip(sets, per) if s[4:] not in NOIMNET])))
    print(tag, len(sets), 'sets', f'{time.time() - t0:.0f}s', flush=True)
    return res


def pick(cands, R, key='mean', tie=0.1):
    six = {c: float(np.mean([R[t][c][key] for t in TAGS])) for c, _ in cands}
    worst = {c: float(min(R[t][c][key] for t in TAGS)) for c, _ in cands}
    npass = dict(cands); top = max(six.values()); tied = [c for c in six if six[c] >= top - tie]
    def phi_of(c): return PHI.index(c[0][1])
    def kk_pref(c): return 0 if c[0][4] == 10 else 1
    best = min(tied, key=lambda c: (npass[c], -worst[c], len(c), phi_of(c), kk_pref(c), -six[c]))
    return best, tied, six, worst


if __name__ == '__main__':
    cands = candidates(4)
    R = {t: tier_scores(t, cands) for t in TAGS}
    best, tied, six, worst = pick(cands, R)
    best2, tied2, six2, worst2 = pick(cands, R, 'mean_noimnet')
    knn = (('C1', 'g', 'L1', 0, 10),)
    order = sorted(six, key=lambda c: -six[c])
    print('\nkNN (PROTOCOL_CT choice):', round(six[knn], 2), 'worst', round(worst[knn], 2), [round(R[t][knn]['mean'], 1) for t in TAGS])
    print('top 15:')
    for c in order[:15]: print(f'  {six[c]:6.2f} worst {worst[c]:5.1f} passes {dict(cands)[c]}  ' + ' '.join(f'{R[t][c]["mean"]:5.1f}' for t in TAGS) + f'  {c}')
    print('tied within 0.1:', len(tied)); print('CHOICE:', best, round(six[best], 2), 'passes', dict(cands)[best])
    print('check (no ImageNet-derived sets): choice', best2, round(six2[best2], 2))
    out = dict(protocol_sha256=hashlib.sha256(open(f'{SCR}/final/eu_stage/push_k/PROTOCOL_OK.md', 'rb').read()).hexdigest(),
               choice=[list(map(str, c)) for c in best], choice_passes=dict(cands)[best], choice_mean=six[best], choice_worst=worst[best],
               tied=[[list(map(str, c)) for c in x] for x in tied], knn_mean=six[knn], knn_worst=worst[knn],
               check_noimnet=dict(choice=[list(map(str, c)) for c in best2], mean=six2[best2]),
               all=[dict(c=[list(map(str, x)) for x in c], passes=dict(cands)[c], mean=six[c], worst=worst[c],
                         per_id={t: R[t][c]['mean'] for t in TAGS}) for c in order])
    json.dump(out, open('ok_select.json', 'w'), indent=0)
