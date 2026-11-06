# insert_e8.py -- inserts the E8 control jobs (one per ID tier) into runs/jobs/Q_hires.txt before the first unclaimed line that is
#   not an E5B/E6/E7 job (i.e. right after the still-pending E5B/E7 jobs); only unclaimed lines move; atomic replace
import os, shutil
Q = 'runs/jobs/Q_hires.txt'; C = 'runs/jobs/Q_hires_claims'
lines = open(Q).read().split('\n')
if lines and lines[-1] == '': lines = lines[:-1]
claimed = sorted(int(f) for f in os.listdir(C))
keys = ('e5b_tier.sh', 'two_controls.py', 'prep_c112.py', 'hires2_seed.sh', 'hires2_cfs.sh')
pos = next(i for i, l in enumerate(lines) if i not in claimed and not any(k in l for k in keys))
assert max(claimed) < pos, ('a claim lies at or after the insertion point', max(claimed), pos)
assert not any('two_controls2.py' in l for l in lines), 'already inserted'
new = [f'python code/two_controls2.py {t} {{gpu}} > logs/e6/two_controls2_{t}.log 2>&1' for t in ('in64', 'edmc10', 'edmc100', 'l6svhn', 'dpceleba', 'fmnist')]
shutil.copy(Q, Q + '.bak_e8')
out = lines[:pos] + new + lines[pos:]
open(Q + '.tmp', 'w').write('\n'.join(out) + '\n'); os.replace(Q + '.tmp', Q)
print('inserted', len(new), 'jobs at', pos, '; total', len(out), '; max claimed', max(claimed))
