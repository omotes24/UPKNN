# move_csi.py -- moves the five CSI jobs of runs/jobs/Q_hires.txt in front of the E4 block (they take ~40 GPU-h each, E4/E5 only hours);
#   only unclaimed lines move (all of them lie after the claimed lines); atomic replace
import os, shutil
Q = 'runs/jobs/Q_hires.txt'; C = 'runs/jobs/Q_hires_claims'
lines = open(Q).read().split('\n')
if lines and lines[-1] == '': lines = lines[:-1]
csi = [i for i, l in enumerate(lines) if 'csi_run.py' in l]; e4 = min(i for i, l in enumerate(lines) if 'e4_tier.sh' in l)
assert len(csi) == 5 and min(csi) > e4, (csi, e4)
claimed = sorted(int(f) for f in os.listdir(C)); assert max(claimed) < e4 - 2, ('claims too close', max(claimed), e4)
rest = [l for i, l in enumerate(lines) if i not in csi]; out = rest[:e4] + [lines[i] for i in csi] + rest[e4:]
assert sorted(out) == sorted(lines) and len(out) == len(lines)
shutil.copy(Q, Q + '.bak_csimove')
claimed2 = sorted(int(f) for f in os.listdir(C)); assert max(claimed2) < e4 - 2
open(Q + '.tmp', 'w').write('\n'.join(out) + '\n'); os.replace(Q + '.tmp', Q)
print('moved CSI jobs to', e4, '-', e4 + 4, '; claimed', max(claimed2))
