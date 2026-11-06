# insert_e9.py -- inserts the E9 jobs (code/e9_space.py, one per ID tier; a minute each) into runs/jobs/Q_hires.txt right after
#   the highest claimed line, so that they run before the remaining LMD / LID+NLL jobs; only unclaimed lines move; atomic replace
import os, shutil
Q = 'runs/jobs/Q_hires.txt'; C = 'runs/jobs/Q_hires_claims'
lines = open(Q).read().split('\n')
if lines and lines[-1] == '': lines = lines[:-1]
claimed = sorted(int(f) for f in os.listdir(C)); pos = max(claimed) + 1
assert not any('e9_space.py' in l for l in lines), 'already inserted'
os.makedirs('logs/e9', exist_ok=True)
new = [f'python code/e9_space.py {t} {{gpu}} > logs/e9/e9_space_{t}.log 2>&1' for t in ('edmc10', 'l6svhn', 'dpceleba', 'edmc100', 'in64', 'fmnist')]
shutil.copy(Q, Q + '.bak_e9')
out = lines[:pos] + new + lines[pos:]
open(Q + '.tmp', 'w').write('\n'.join(out) + '\n'); os.replace(Q + '.tmp', Q)
print('inserted', len(new), 'jobs at', pos, '; total', len(out), '; max claimed', max(claimed))
