# export_baselines.py -- per-image scores of the eight comparison methods on the common images (from analysis/baselines_eval.py,
#   unchanged fits), cached for the bootstrap analyses: baselines_perimage.pkl = {tag: {set ('__id__' | OOD name): {method: array}}}
import pickle, json, sys, numpy as np
from ana_common import *
import baselines_eval as BE
out = {}
for tag, T in TIERS.items():
    S, info = BE.tier_scores(tag, T)
    d = {}
    for s in ['__id__'] + T['ood']:
        name = setname(tag, s)
        d[s] = {m: sub(*S[name][m], COM[tag][s]) for m in BASELINES if m in S[name]}
    out[tag] = d
    # consistency with results/baselines_pairs.json
    B = json.load(open(f'{REPO}/results/baselines_pairs.json'))[tag]['common']
    bad = [(m, s) for m in BASELINES for s in T['ood'] if s in B[m] and abs(100 * auroc(d['__id__'][m], d[s][m]) - 100 * B[m][s]['auroc']) > 1e-6]
    print(tag, 'mismatches vs baselines_pairs.json:', bad[:5], flush=True)
pickle.dump(out, open('baselines_perimage.pkl', 'wb'))
print('saved')
