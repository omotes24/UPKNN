# insert_e6.py -- inserts the E6 control jobs (one line per ID tier) into runs/jobs/Q_hires.txt right after the E5B lines (before the
#   first unclaimed kamkari_lid.py line); only unclaimed lines move; atomic replace (claims are by line index)
import os, shutil
Q = 'runs/jobs/Q_hires.txt'; C = 'runs/jobs/Q_hires_claims'
lines = open(Q).read().split('\n')
if lines and lines[-1] == '': lines = lines[:-1]
claimed = sorted(int(f) for f in os.listdir(C))
pos = next(i for i, l in enumerate(lines) if 'kamkari_lid.py' in l and i not in claimed)
assert max(claimed) < pos, ('a claim lies at or after the insertion point', max(claimed), pos)
assert not any('two_controls.py' in l for l in lines), 'already inserted'
new = [f'python code/two_controls.py {t} {{gpu}} > logs/e6/two_controls_{t}.log 2>&1' for t in ('in64', 'edmc10', 'edmc100', 'l6svhn', 'dpceleba', 'fmnist')]
shutil.copy(Q, Q + '.bak_e6')
out = lines[:pos] + new + lines[pos:]
open(Q + '.tmp', 'w').write('\n'.join(out) + '\n'); os.replace(Q + '.tmp', Q)
print('inserted', len(new), 'jobs at', pos, '; total', len(out), '; max claimed', max(claimed))
