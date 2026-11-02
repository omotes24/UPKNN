# ok_prof_sel.py -- extension 3 (selection data only): AUROC of the similarity-profile statistics (r-th largest cosine and mean of
#   the r largest, r up to 1000; spaces L1 / L012; sigma_1 / sigma_7)
import json, numpy as np
from ok_common import *
ND = f'{SCR}/final/data_ok'
def sets(tag): return [r[0] for r in json.load(open(f'{ND}/sets_{tag}.json')) if r[0].startswith('sel-')]
R = {}
for tag in TAGS:
    names = ['devall-' + IDN[tag]] + sets(tag)
    Z = {n: np.load(f'{ND}/prof/s0/{tag}__{n}.npz') for n in names}; RS = list(Z[names[0]]['R'])
    for key in [k for k in Z[names[0]].files if k != 'R']:
        nr = Z[names[0]][key].shape[1]
        for j in range(nr):
            nm = f'{key}_r{RS[j]}'
            R.setdefault(nm, {})[tag] = float(np.mean([100 * auroc(Z[names[0]][key][:, j], Z[n][key][:, j]) for n in names[1:]]))
six = {k: np.mean([v.get(t, np.nan) for t in TAGS]) for k, v in R.items() if len(v) == 6}
for k in sorted(six, key=lambda k: -six[k])[:30]: print(f'{k:20s} {six[k]:6.2f} ' + ' '.join(f'{R[k][t]:5.1f}' for t in TAGS))
print('in64 bank is 10k; r=1000 is 10% of it')
json.dump(R, open('ok_prof_sel.json', 'w'))
