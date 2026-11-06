# csi_ana.py -- CSI (runs/kx/PROTOCOL_CSI.md; official code, one training run per ID) against Upright kNN on the final2 images of the
#   five 32x32 IDs, with the bootstrap of ours_vs_all.py: 2000 replicates, rng 20261031 and the same draw order over all six tiers (ID
#   images once per ID, OOD images per set), so the resampled images are those of ours_vs_all_final2.json (checked on the RotCls4 row);
#   Upright kNN averaged over noise seeds 0-2; win / loss if the 95% interval of AUROC(Ours) - AUROC(CSI) excludes 0 and |gap| >= 0.05.
#   Written before any CSI score was read (2026-11-02).
#   inputs: ../res/scores/<tag>/final2-<set>.npz  (runs/csi/scores/<tag>/ of the GPU server: score [higher = ID], idx)
#           ../res/logs/<tag>/log.txt, SCORE_DONE (runs/csi/logs/<run>/log.txt, runs/csi/scores/<tag>/SCORE_DONE): cost
#   output: csi_ana.json
import os, sys, json, re, datetime
import numpy as np
os.environ['SPLIT'] = 'final2'
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.environ.get('CSI_RES', os.path.join(HERE, '..', 'res'))   # CSI_RES: dry runs
SCR = os.environ.get('UPKNN_ANALYSIS', os.path.expanduser('~/upknn/analysis'))
sys.path.insert(0, f'{SCR}/final/ana')
from ana_common import *
from boot import counts, boot_auroc
B = 2000; SEEDS = (0, 1, 2); SPLIT = 'final2'
CSI_TAGS = ['edmc10', 'edmc100', 'l6svhn', 'dpceleba', 'fmnist']
COMX = json.load(open(f'{SCR}/final/eu_stage/push_final2/final2_idx/final2_idx.json'))
EXP = f'{SCR}/final/data_kx/export'
REF = json.load(open(f'{SCR}/final/ana_kx/ours_vs_all_final2.json'))
PUBLISHED_C10_C100 = 89.2                                              # Tack et al. 2020, Table 2(a): unlabeled multi-class CIFAR-10 -> CIFAR-100, 89.2 +- 0.1


def verdict(lo, hi, gap):
    if abs(gap) < 0.05: return 'tie'
    return 'win' if lo > 0 else ('loss' if hi < 0 else 'tie')


def nm(tag, s): return setname(tag, s)


def ours(tag, s, seed):
    Z = np.load(f'{EXP}/{tag}_s{seed}_{SPLIT}.npz'); n = f'{SPLIT}-{nm(tag, s)}'
    return sub(Z[f'score__{n}'].astype(np.float64), Z[f'idx__{n}'], COMX[tag][s])


def csi(tag, s):
    z = np.load(f'{RES}/scores/{tag}/{SPLIT}-{nm(tag, s)}.npz'); v = sub(z['score'].astype(np.float64), z['idx'], COMX[tag][s])
    assert len(v) == len(COMX[tag][s]) and np.all(np.isfinite(v)), (tag, s, len(v))
    return v


def rotcls4(tag, s):                                                   # only to check that the bootstrap draws equal ours_vs_all.py's
    z = np.load(f'{DATA}/runs/rotcls/{tag}/scores_{SPLIT}/{nm(tag, s)}.npz'); return sub(z['rot4'].astype(np.float64), z['idx'], COMX[tag][s])


rng = np.random.default_rng(20261031)
OUT = {'split': SPLIT, 'B': B, 'published_c10_c100': PUBLISHED_C10_C100, 'pairs': {}, 'ids': {}, 'check_rotcls4': []}
BO, BC = {}, {}
for tag, T in TIERS.items():
    oi = {sd: ours(tag, '__id__', sd) for sd in SEEDS}; Wa = counts(rng, len(oi[0]), B)
    have = tag in CSI_TAGS
    if have: ci = csi(tag, '__id__'); assert len(ci) == len(oi[0])
    pr = {}
    for o in T['ood']:
        oo = {sd: ours(tag, o, sd) for sd in SEEDS}; Wb = counts(rng, len(oo[0]), B)
        ob = np.mean([boot_auroc(oi[sd], oo[sd], Wa, Wb) for sd in SEEDS], 0); opt = [100 * auroc(oi[sd], oo[sd]) for sd in SEEDS]
        ref = REF['pairs'][tag][o]
        assert abs(np.mean(opt) - ref['ours']) < 1e-9, (tag, o, np.mean(opt), ref['ours'])
        if 'RotCls4' in ref['rows']:                                    # same draws as ours_vs_all.py -> same interval for its RotCls4 row
            rb = boot_auroc(rotcls4(tag, '__id__'), rotcls4(tag, o), Wa, Wb); lo, hi = np.percentile(ob - rb, [2.5, 97.5])
            OUT['check_rotcls4'].append(max(abs(lo - ref['rows']['RotCls4']['ci'][0]), abs(hi - ref['rows']['RotCls4']['ci'][1])))
        if not have: continue
        co = csi(tag, o); assert len(co) == len(oo[0])
        cb = boot_auroc(ci, co, Wa, Wb); cp = 100 * auroc(ci, co)
        lo, hi = np.percentile(ob - cb, [2.5, 97.5]); gap = float(np.mean(opt) - cp)
        pr[o] = dict(ours=float(np.mean(opt)), csi=float(cp), diff=gap, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, gap),
                     ours_fpr95=float(np.mean([100 * fpr95(oi[sd], oo[sd]) for sd in SEEDS])), csi_fpr95=float(100 * fpr95(ci, co)),
                     kind='natural' if o in NAT[tag] else ('upside-down' if o.endswith('__vflip') else 'synthetic'))
        BO[(tag, o)], BC[(tag, o)] = ob, cb
    if not have: continue
    OUT['pairs'][tag] = pr
    nat = NAT[tag]; om = float(np.mean([pr[o]['ours'] for o in nat])); cm = float(np.mean([pr[o]['csi'] for o in nat]))
    dn = np.mean([BO[(tag, o)] for o in nat], 0) - np.mean([BC[(tag, o)] for o in nat], 0); lo, hi = np.percentile(dn, [2.5, 97.5])
    OUT['ids'][tag] = dict(name=ID_NAME[tag], ours_nat=om, csi_nat=cm, diff=om - cm, ci=[float(lo), float(hi)], verdict=verdict(lo, hi, om - cm),
                           csi_fpr95_nat=float(np.mean([pr[o]['csi_fpr95'] for o in nat])), ours_fpr95_nat=float(np.mean([pr[o]['ours_fpr95'] for o in nat])))
    print(f"{ID_NAME[tag]:13s} natural mean  Upright kNN {om:5.1f}  CSI {cm:5.1f}  diff {om - cm:+5.1f} [{lo:+.1f}, {hi:+.1f}]", flush=True)
assert max(OUT['check_rotcls4']) < 1e-9, max(OUT['check_rotcls4'])


def tally(sel):
    vs = [p['verdict'] for t in OUT['pairs'] for o, p in OUT['pairs'][t].items() if sel(t, o, p)]
    return dict(wins=vs.count('win'), ties=vs.count('tie'), losses=vs.count('loss'), pairs=len(vs))


OUT['wtl_all'] = tally(lambda t, o, p: True); OUT['wtl_nat'] = tally(lambda t, o, p: p['kind'] == 'natural')
OUT['upside_down'] = {t: {k: OUT['pairs'][t][o][k] for k in ('ours', 'csi', 'diff', 'ci', 'verdict')} for t in OUT['pairs'] for o in OUT['pairs'][t] if o.endswith('__vflip')}
OUT['mean_nat'] = dict(ours=float(np.mean([v['ours_nat'] for v in OUT['ids'].values()])), csi=float(np.mean([v['csi_nat'] for v in OUT['ids'].values()])))
if 'edmc10' in OUT['pairs']: OUT['repro_c10_c100'] = OUT['pairs']['edmc10']['cifar100_test']['csi']


def train_hours(path):
    """wall-clock of the epochs in log.txt: differences between consecutive '[DONE]' lines (gaps above 1 h, i.e. interruptions, dropped)
    plus one median epoch for the first"""
    ts = [datetime.datetime.fromisoformat(m.group(1)) for m in re.finditer(r'^\[([0-9-]+ [0-9:.]+)\] \[DONE\]', open(path).read(), re.M)]
    d = np.diff([t.timestamp() for t in ts]); d = d[d < 3600]
    return dict(epochs_logged=len(ts), hours=float((d.sum() + np.median(d)) / 3600), sec_per_epoch=float(np.median(d)))


OUT['cost'] = {}
for t in OUT['pairs']:
    c = {}
    if os.path.exists(f'{RES}/logs/{t}/log.txt'): c.update(train_hours(f'{RES}/logs/{t}/log.txt'))
    if os.path.exists(f'{RES}/logs/{t}/SCORE_DONE'):
        sec = float(open(f'{RES}/logs/{t}/SCORE_DONE').read().split()[1]); n = sum(len(np.load(f'{RES}/scores/{t}/{f}')['score']) for f in os.listdir(f'{RES}/scores/{t}') if f.endswith('.npz'))
        c.update(score_seconds=sec, test_images=n)                       # includes the 40 passes over each of the 50,000 training images
    OUT['cost'][t] = c
print('W/T/L all', OUT['wtl_all'], ' natural', OUT['wtl_nat'], ' mean natural', {k: round(v, 2) for k, v in OUT['mean_nat'].items()})
print('CIFAR-10 -> CIFAR-100: CSI', round(OUT.get('repro_c10_c100', float('nan')), 1), 'published', PUBLISHED_C10_C100)
json.dump(OUT, open(os.path.join(HERE, 'csi_ana.json' if 'CSI_RES' not in os.environ else 'csi_ana_dryrun.json'), 'w'), indent=1)
