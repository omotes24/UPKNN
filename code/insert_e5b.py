# insert_e5b.py -- inserts the two E5B tier jobs into runs/jobs/Q_hires.txt right before the first unclaimed kamkari_lid.py line
#   (after the LMD block); only unclaimed lines move; atomic replace (workers re-read the file on every scan; claims are by line index)
import os, shutil
Q = 'runs/jobs/Q_hires.txt'; C = 'runs/jobs/Q_hires_claims'
lines = open(Q).read().split('\n')
if lines and lines[-1] == '': lines = lines[:-1]
claimed = sorted(int(f) for f in os.listdir(C))
pos = next(i for i, l in enumerate(lines) if 'kamkari_lid.py' in l and i not in claimed)
assert max(claimed) < pos, ('a claim lies at or after the insertion point', max(claimed), pos)
assert not any('e5b_tier.sh' in l for l in lines), 'already inserted'
new = [f'bash code/e5b_tier.sh {t} {{gpu}}' for t in ('in64b50', 'in64b25')]
shutil.copy(Q, Q + '.bak_e5b')
out = lines[:pos] + new + lines[pos:]
open(Q + '.tmp', 'w').write('\n'.join(out) + '\n'); os.replace(Q + '.tmp', Q)
print('inserted', len(new), 'jobs at', pos, '; total', len(out), '; max claimed', max(claimed))
