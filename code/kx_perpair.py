# kx_perpair.py -- DEVELOPMENT: per-pair fresh AUROC of given spaces (statistic chosen on selection per ID, as kx_ana2.py)
import sys, numpy as np
sys.argv = [sys.argv[0], sys.argv[1], '']
exec(open('code/kx_ana2.py').read().split('BEST = {}')[0])
SPS = sys.argv[1].split(',')
for t in TAGS:
    print('==', t)
    rows = {}
    for sp in SPS:
        sc = {j: sel_auc(t, lambda n, j=j: stat(t, sp, n, j)) for j in stats(t, sp)}; j = max(sc, key=sc.get)
        rows[sp] = (sname(t, sp, j), fresh_aucs(t, lambda n: stat(t, sp, n, j)))
    print(f"{'set':26s} " + ' '.join(f'{sp[:14]:>14s}' for sp in SPS))
    print(f"{'(stat)':26s} " + ' '.join(f'{rows[sp][0]:>14s}' for sp in SPS))
    for o in NAT[t] + [VF[t]]:
        print(f'{o:26s} ' + ' '.join(f'{rows[sp][1][o]:14.1f}' for sp in SPS))
