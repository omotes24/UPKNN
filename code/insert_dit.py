# insert_dit.py -- inserts the three dit256 post-extraction jobs (hires_post_dit.sh, seeds 0-2) into runs/jobs/Q_hires.txt just after the
#   claimed lines (+2), so that they run next; only unclaimed lines move; atomic replace
import os, shutil
Q = 'runs/jobs/Q_hires.txt'; C = 'runs/jobs/Q_hires_claims'
lines = open(Q).read().split('\n')
if lines and lines[-1] == '': lines = lines[:-1]
assert not any('hires_post_dit.sh' in l for l in lines), 'already inserted'
claimed = sorted(int(f) for f in os.listdir(C)); POS = max(claimed) + 3
new = [f'bash code/hires_post_dit.sh {{gpu}} {s} > logs/hires/post_dit256_{s}.log 2>&1' for s in (0, 1, 2)]
shutil.copy(Q, Q + '.bak_dit')
out = lines[:POS] + new + lines[POS:]
claimed2 = sorted(int(f) for f in os.listdir(C)); assert max(claimed2) < POS, ('claimed meanwhile', max(claimed2), POS)
open(Q + '.tmp', 'w').write('\n'.join(out) + '\n'); os.replace(Q + '.tmp', Q)
print('inserted', len(new), 'jobs at', POS, '; total', len(out), '; claimed', max(claimed2))
