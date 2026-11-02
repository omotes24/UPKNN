# insert_csi.py -- inserts the CSI jobs (PROTOCOL_CSI.md) into runs/jobs/Q_hires.txt after the E5 block (before the 6-ID
#   Graham/LMD/Kamkari block); only unclaimed lines move; atomic replace (workers re-read the file on every scan; claims are by line index)
import os, shutil
Q = 'runs/jobs/Q_hires.txt'; C = 'runs/jobs/Q_hires_claims'
lines = open(Q).read().split('\n')
if lines and lines[-1] == '': lines = lines[:-1]
assert not any('csi_run.py' in l for l in lines), 'already inserted'
POS = max(i for i, l in enumerate(lines) if 'e5_tier.sh' in l) + 1
claimed = sorted(int(f) for f in os.listdir(C))
assert max(claimed) < POS - 10, ('claims too close', max(claimed), POS)
assert 'graham.py' in lines[POS] and 'e5_tier.sh' in lines[POS - 1], (lines[POS - 1][:80], lines[POS][:80])
new = [f'python code/csi_run.py train {t} {{gpu}} && python code/csi_run.py score {t} {{gpu}} final2-'
       for t in ('edmc10', 'edmc100', 'l6svhn', 'dpceleba', 'fmnist')]
shutil.copy(Q, Q + '.bak_csi')
out = lines[:POS] + new + lines[POS:]
open(Q + '.tmp', 'w').write('\n'.join(out) + '\n'); os.replace(Q + '.tmp', Q)
print('inserted', len(new), 'jobs at', POS, '; total', len(out), '; claimed so far', claimed[-3:], len(claimed))
