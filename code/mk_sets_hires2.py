# mk_sets_hires2.py -- runs/ok/sets_in256d.json (prefix q256d-, data hc112_*) for PROTOCOL_HIRES2.md: the sets of sets_in256c.json
#   (same images, same order) on the identically resampled arrays.
import os, json
ROOT = os.path.expanduser('~/upknn'); OUT = f'{ROOT}/runs/ok/sets_in256d.json'
assert not os.path.exists(OUT), 'exists'
S = json.load(open(f'{ROOT}/runs/ok/sets_in256c.json'))
T = [[n.replace('q256-', 'q256d-', 1), d.replace('hc_', 'hc112_', 1), idx, pre] for n, d, idx, pre in S]
assert all(n.startswith('q256d-') and d.startswith('hc112_') for n, d, _, _ in T)
json.dump(T, open(OUT, 'w')); print([(r[0], r[1], len(r[2]), r[3]) for r in T])
