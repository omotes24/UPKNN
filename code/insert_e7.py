# insert_e7.py -- inserts the PROTOCOL_HIRES2.md jobs into runs/jobs/Q_hires.txt right before the first unclaimed kamkari_lid.py line
#   (after the E5B and E6 lines); only unclaimed lines move; atomic replace (claims are by line index)
import os, shutil
Q = 'runs/jobs/Q_hires.txt'; C = 'runs/jobs/Q_hires_claims'
lines = open(Q).read().split('\n')
if lines and lines[-1] == '': lines = lines[:-1]
claimed = sorted(int(f) for f in os.listdir(C))
pos = next(i for i, l in enumerate(lines) if 'kamkari_lid.py' in l and i not in claimed)
assert max(claimed) < pos, ('a claim lies at or after the insertion point', max(claimed), pos)
assert not any('hires2_seed.sh' in l for l in lines), 'already inserted'
new = ['python code/prep_c112.py > logs/hires2/prep.log 2>&1 && python code/mk_sets_hires2.py >> logs/hires2/prep.log 2>&1 && touch logs/hires2/PREP_DONE']
new += [f'bash code/hires2_seed.sh {{gpu}} {s} > logs/hires2/seed_{s}.log 2>&1' for s in (0, 1, 2)]
new += [f'bash code/hires2_cfs.sh {{gpu}} {s} > logs/hires2/cfs_{s}.log 2>&1' for s in (0, 1, 2)]
shutil.copy(Q, Q + '.bak_e7')
out = lines[:pos] + new + lines[pos:]
open(Q + '.tmp', 'w').write('\n'.join(out) + '\n'); os.replace(Q + '.tmp', Q)
print('inserted', len(new), 'jobs at', pos, '; total', len(out), '; max claimed', max(claimed))
