# fiveid.py -- the five ID datasets of the paper.  The evaluation files in gen/data/ were written for six ID datasets; the sixth, CIFAR-100
#   with the third-party checkpoint edm_cifar100_uncond_vp.pt, was removed after the final evaluation when the checkpoint turned out to be a
#   CIFAR-10 model (protocols/CIFAR100_CHECKPOINT.md).  The result files are kept as they are (they are the record of the frozen evaluation);
#   the functions below drop the CIFAR-100 tier from a loaded file and recompute every aggregate (means over ID datasets, wins / ties /
#   losses, near-OOD means) from the per-dataset and per-pair entries that the files already hold, with the same definitions as the scripts
#   that wrote them.  Nothing is re-evaluated and no pair of the remaining five datasets changes.
import numpy as np
DROP = 'edmc100'
TAGS = ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']
NEAR = [('edmc10', 'cifar100_test'), ('in64', 'in64_ssb_hard'), ('in64', 'in64_ninco')]      # the near-OOD pairs (CIFAR-100 -> CIFAR-10 is gone)
NEAR_KEYS = [f'{t}/{o}' for t, o in NEAR]


def is_nat(o): return not o.endswith('__vflip') and 'synth' not in o


def _mean(v): return float(np.mean(v))


def _wtl(verdicts, win='wins', tie='ties', loss='losses'):
    return {win: verdicts.count('win'), tie: verdicts.count('tie'), loss: verdicts.count('loss')}


def J5(J):
    """ours_vs_all_*.json: pairs / ids per tag, rows totals, mean_ours."""
    for k in ('pairs', 'ids'): J[k].pop(DROP, None)
    tags = [t for t in TAGS if t in J['ids']]
    J['mean_ours'] = _mean([J['ids'][t]['ours'] for t in tags])
    for r in J['rows']:
        vs = [p['rows'][r]['verdict'] for t in tags for p in J['pairs'][t].values() if r in p['rows']]
        J['rows'][r].update(_wtl(vs)); J['rows'][r]['pairs'] = len(vs)
        J['rows'][r]['mean_nat'] = _mean([J['ids'][t]['rows'][r]['mean'] for t in tags if r in J['ids'][t]['rows']])
    return J


def NEWJ5(N):
    """newrows_*.json (DDPM-OOD, LMD, LID+NLL): the same, with the natural-pair counts and the tiers of each row."""
    for k in ('pairs', 'ids'): N[k].pop(DROP, None)
    tags = [t for t in TAGS if t in N['ids']]
    for r in N['rows']:
        tiers = [t for t in tags if r in N['ids'][t]['rows']]
        vs = [p['rows'][r]['verdict'] for t in tiers for p in N['pairs'][t].values() if r in p['rows']]
        vn = [p['rows'][r]['verdict'] for t in tiers for o, p in N['pairs'][t].items() if is_nat(o) and r in p['rows']]
        N['rows'][r].update(_wtl(vs)); N['rows'][r]['pairs'] = len(vs)
        N['rows'][r].update(_wtl(vn, 'nat_wins', 'nat_ties', 'nat_losses')); N['rows'][r]['nat_pairs'] = len(vn)
        N['rows'][r]['mean_nat'] = _mean([N['ids'][t]['rows'][r]['mean'] for t in tiers]); N['rows'][r]['tiers'] = tiers
    return N


def FP5(FP):
    """fpr95_all_*.json: mean (fpr95, auroc) and vflip_fpr95 per row over the remaining datasets."""
    for k in ('pairs', 'ids'): FP[k].pop(DROP, None)
    tags = [t for t in TAGS if t in FP['ids']]
    for r in FP['mean']:
        FP['mean'][r] = {k: _mean([FP['ids'][t][r][k] for t in tags if r in FP['ids'][t]]) for k in ('fpr95', 'auroc')}
    for r in FP['vflip_fpr95']:
        FP['vflip_fpr95'][r] = _mean([FP['pairs'][t][o][r]['fpr95'] for t in tags for o in FP['pairs'][t] if o.endswith('__vflip') and r in FP['pairs'][t][o]])
    return FP


def CSI5(C):
    """csi_ana.json: the four remaining 32x32 datasets."""
    for k in ('pairs', 'ids', 'upside_down', 'cost'): C[k].pop(DROP, None)
    tags = list(C['ids'])
    C['mean_nat'] = {'ours': _mean([C['ids'][t]['ours_nat'] for t in tags]), 'csi': _mean([C['ids'][t]['csi_nat'] for t in tags])}
    nat = [p['verdict'] for t in tags for p in C['pairs'][t].values() if p['kind'] == 'natural']
    allp = [p['verdict'] for t in tags for p in C['pairs'][t].values()]
    C['wtl_nat'] = dict(_wtl(nat), pairs=len(nat)); C['wtl_all'] = dict(_wtl(allp), pairs=len(allp))
    return C


def DEEP5(Dp):
    """vl_vs_uknn_*.json: wtl, mean_vl, mean_upright."""
    for k in ('pairs', 'ids'): Dp[k].pop(DROP, None)
    tags = [t for t in TAGS if t in Dp['ids']]
    vs = [q['verdict'] for t in tags for q in Dp['pairs'][t].values()]
    Dp['wtl'] = dict(_wtl(vs), pairs=len(vs))
    Dp['mean_vl'] = _mean([Dp['ids'][t]['vl'] for t in tags]); Dp['mean_upright'] = _mean([Dp['ids'][t]['upright'] for t in tags])
    return Dp


def DEEPC5(Dc):
    """vl_ctrl_fresh.json: mean per rule, wtl (vs Upright) and wtl_vs_vl."""
    for k in ('pairs', 'ids', 'choices'): Dc[k].pop(DROP, None)
    tags = [t for t in TAGS if t in Dc['ids']]
    for r in Dc['mean']: Dc['mean'][r] = _mean([Dc['ids'][t][r] for t in tags])
    for key, vs_ in (('wtl', 'vs_Upright'), ('wtl_vs_vl', 'vs_VL')):
        for r in Dc[key]:
            vs = [q[r][vs_]['verdict'] for t in tags for q in Dc['pairs'][t].values() if r in q and vs_ in q[r]]
            Dc[key][r] = dict(_wtl(vs), pairs=len(vs))
    return Dc


def KG5(KG):
    """knn_grid.json: mean6 (now the five-dataset mean, the key is kept), vflip and near of every configuration and head, both splits."""
    for split in KG['grid']:
        for coll in (KG['grid'][split], KG['heads'][split]):
            for c in coll.values():
                c['ids'].pop(DROP, None); c['pairs'].pop(DROP, None); tags = [t for t in TAGS if t in c['ids']]
                c['mean6'] = _mean([c['ids'][t] for t in tags])
                c['vflip'] = _mean([v for t in tags for o, v in c['pairs'][t].items() if o.endswith('__vflip')])
                c['near'] = _mean([c['pairs'][t][o] for t, o in NEAR if t in c['pairs'] and o in c['pairs'][t]])
    return KG


def F5(F):
    """fusion_final2.json: mean, vflip_mean, wtl_nat, wtl_all per fusion."""
    for k in ('ids', 'pairs', 'vflip'): F[k].pop(DROP, None)
    tags = [t for t in TAGS if t in F['ids']]
    for r in F['mean']: F['mean'][r] = _mean([F['ids'][t][r] for t in tags])
    for r in F['vflip_mean']: F['vflip_mean'][r] = _mean([F['vflip'][t][r] for t in tags])
    for key, natonly in (('wtl_nat', True), ('wtl_all', False)):
        for r in F[key]:
            vs = [p[r]['verdict'] for t in tags for o, p in F['pairs'][t].items() if (is_nat(o) or not natonly) and r in p and 'verdict' in p[r]]
            F[key][r] = _wtl(vs, 'win', 'tie', 'loss')
    return F


def WS5(W):
    """weight_sweep.json: per split, mean_nat / mean_vflip / near_mean over the weights, argmax_w."""
    for split in W:
        S = W[split]; S['ids'].pop(DROP, None); S['argmax_w_per_id'].pop(DROP, None)
        S['pairs'] = {k: v for k, v in S['pairs'].items() if not k.startswith(DROP + '/')}
        tags = [t for t in TAGS if t in S['ids']]; nw = len(S['w'])
        S['mean_nat'] = [_mean([S['ids'][t]['nat'][i] for t in tags]) for i in range(nw)]
        S['mean_vflip'] = [_mean([S['ids'][t]['vflip'][i] for t in tags]) for i in range(nw)]
        S['near_mean'] = [_mean([S['pairs'][k][i] for k in NEAR_KEYS if k in S['pairs']]) for i in range(nw)]
        S['argmax_w'] = S['w'][int(np.argmax(S['mean_nat']))]
    return W


def E95(E):
    """e9_ana.json: mean, vflip_mean, mean_tiers, wtl_nat, big_nat per fusion."""
    for k in ('pairs', 'ids', 'vflip', 'fusions'): E[k].pop(DROP, None)
    tags = [t for t in TAGS if t in E['ids']]
    for r in E['mean']:
        tiers = [t for t in tags if r in E['ids'][t]]; E['mean_tiers'][r] = tiers; E['mean'][r] = _mean([E['ids'][t][r] for t in tiers])
    for r in E['vflip_mean']: E['vflip_mean'][r] = _mean([E['vflip'][t][r] for t in tags if r in E['vflip'][t]])
    for k in E['wtl_nat']:
        vs = [p['_vs'][k] for t in tags for o, p in E['pairs'][t].items() if is_nat(o) and k in p.get('_vs', {})]
        E['wtl_nat'][k] = _wtl([v['verdict'] for v in vs], 'win', 'tie', 'loss')
        E['big_nat'][k] = {'win1': sum(v['verdict'] == 'win' and v['diff'] >= 1 for v in vs), 'loss1': sum(v['verdict'] == 'loss' and v['diff'] <= -1 for v in vs)}
    return E


def drop_tag(d):
    """a dict keyed by tag (ablate2, dR, two_notions, ...): the same dict without the CIFAR-100 entry."""
    if isinstance(d, dict): d.pop(DROP, None)
    return d
