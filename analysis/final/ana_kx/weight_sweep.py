# weight_sweep.py -- POST-HOC (no change to the method): the relative weight of the two terms, s_w = -d log r_R + w log q(R0|x), on the
#   development images (split 'fresh') and, for reference, on the final evaluation images; w = 1 is the paper's score.  Because the kNN
#   term is -d log r_R, a different estimate d' of the intrinsic dimension is the same as w = d / d', so the sweep also shows how much an
#   error in d would matter.  Mean natural-OOD AUROC per ID dataset (seeds 0-2), the upside-down probe, and the near-OOD pairs.
#   usage: python weight_sweep.py   -> weight_sweep.json
import os, sys, json
import numpy as np
from scipy.stats import rankdata
SCR = os.environ.get('UPKNN_ANALYSIS', os.path.expanduser('~/upknn/analysis'))
sys.path.insert(0, f'{SCR}/final/ana'); os.environ['SPLIT'] = 'final2'
from ana_common import TIERS, NAT, ID_NAME, setname
EXP = f'{SCR}/final/data_kx/export'; SEEDS = (0, 1, 2)
WS = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4]
NEAR = {('edmc10', 'cifar100_test'), ('edmc100', 'cifar10_test'), ('in64', 'in64_ssb_hard'), ('in64', 'in64_ninco')}
COMX = {'fresh': json.load(open(f'{SCR}/final/eu_stage/push/fresh_idx/fresh_idx.json')), 'final2': json.load(open(f'{SCR}/final/eu_stage/push_final2/final2_idx/final2_idx.json'))}


def sub(v, idx, keep):
    pos = {int(i): k for k, i in enumerate(idx)}; return np.asarray([v[pos[int(i)]] for i in keep])


def auroc(a, b):
    r = rankdata(np.concatenate([a, b])); return float(100 * (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


OUT = {}
for split in ('fresh', 'final2'):
    res = {'w': WS, 'ids': {}, 'pairs': {}}
    for tag, T in TIERS.items():
        terms = {}
        for sd in SEEDS:
            Z = np.load(f'{EXP}/{tag}_s{sd}_{split}.npz')
            for s in ['__id__'] + T['ood']:
                n = f'{split}-{setname(tag, s)}'
                if f'content__{n}' not in Z.files: continue
                terms[(sd, s)] = (sub(Z[f'content__{n}'].astype(np.float64), Z[f'idx__{n}'], COMX[split][tag][s]),
                                  sub(Z[f'pose__{n}'].astype(np.float64), Z[f'idx__{n}'], COMX[split][tag][s]))
        sets = [s for s in T['ood'] if (0, s) in terms]
        per = {s: [] for s in sets}
        for w in WS:
            for s in sets:
                per[s].append(float(np.mean([auroc(terms[(sd, '__id__')][0] + w * terms[(sd, '__id__')][1], terms[(sd, s)][0] + w * terms[(sd, s)][1]) for sd in SEEDS])))
        nat = [s for s in NAT[tag] if s in sets]; vf = [s for s in sets if s.endswith('__vflip')]
        res['ids'][tag] = dict(name=ID_NAME[tag], nat=[float(np.mean([per[s][k] for s in nat])) for k in range(len(WS))],
                               vflip=[per[vf[0]][k] for k in range(len(WS))] if vf else None, n_nat=len(nat))
        for s in sets: res['pairs'][f'{tag}/{s}'] = per[s]
    res['mean_nat'] = [float(np.mean([res['ids'][t]['nat'][k] for t in TIERS])) for k in range(len(WS))]
    res['mean_vflip'] = [float(np.mean([res['ids'][t]['vflip'][k] for t in TIERS])) for k in range(len(WS))]
    res['near_mean'] = [float(np.mean([res['pairs'][f'{t}/{s}'][k] for t, s in NEAR if f'{t}/{s}' in res['pairs']])) for k in range(len(WS))]
    res['argmax_w'] = WS[int(np.argmax(res['mean_nat']))]
    res['argmax_w_per_id'] = {t: WS[int(np.argmax(res['ids'][t]['nat']))] for t in TIERS}
    OUT[split] = res
    print(f'== {split}: w {WS}')
    print('   mean natural', [round(x, 2) for x in res['mean_nat']])
    print('   mean flip   ', [round(x, 1) for x in res['mean_vflip']])
    print('   near-OOD    ', [round(x, 1) for x in res['near_mean']])
    for t in TIERS: print(f"   {ID_NAME[t]:13s}", [round(x, 1) for x in res['ids'][t]['nat']], 'argmax w =', res['argmax_w_per_id'][t])
json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weight_sweep.json'), 'w'), indent=1)
