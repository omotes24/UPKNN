# e4_resume.py <model dir under runs/> <gpu> -- resumes an interrupted run of eu/experiments/train.py with the arguments saved in its
#   latest.pt (train.py restarts from latest.pt by itself; only --gpu is replaced)          (PROTOCOL_E4.md)
import os, sys, subprocess, torch
ROOT = os.path.expanduser('~/upknn'); d, gpu = sys.argv[1], sys.argv[2]
if os.path.exists(f'{ROOT}/runs/{d}/final.pt'): print('final.pt exists', d, flush=True); sys.exit(0)
ck = torch.load(f'{ROOT}/runs/{d}/latest.pt', map_location='cpu', weights_only=False)
a = dict(ck['cfg']['args']); nimg = ck['nimg']; del ck
assert a['out'] == d, (a['out'], d)
a['gpu'] = int(gpu); argv = []
for k, v in a.items(): argv += [f'--{k}', str(v)]
print('resume', d, 'at', nimg, 'args', a, flush=True)
sys.exit(subprocess.call([sys.executable, f'{ROOT}/eu/experiments/train.py'] + argv, env=dict(os.environ, SR_ROOT=ROOT), cwd=ROOT))
