# assemble_rotb.py <tag> <seed> -- e512 (PROTOCOL_HIRES.md): the rotated-bank feature arrays [N, 4, 10C] of kx_rothead2.py from the view-0
#   features of the four rotated banks (rb512-r0/r90/r180/r270, encoded after rotating the pixels; noise never rotated), and the links of
#   the kNN bank (view 0 = r0) in runs/ok.
import os, sys
import numpy as np
from numpy.lib.format import open_memmap
ROOT = os.path.expanduser('~/upknn'); tag, s = sys.argv[1], int(sys.argv[2]); RB = {'e512': 'rb512-', 'dit256': 'rbdit-'}[tag]
D = f'{ROOT}/runs/ok/s{s}/{tag}/k0'; B = f'{ROOT}/runs/kx/rotb/s{s}/{tag}/k0'; os.makedirs(B, exist_ok=True)
for l in range(3):
    if os.path.exists(f'{B}/bank__L{l}.npy'): continue
    V = [np.load(f'{D}/{RB}r{a}__L{l}.npy', mmap_mode='r') for a in (0, 90, 180, 270)]
    N, _, Fd = V[0].shape; out = open_memmap(f'{B}/bank__L{l}.npy.part', 'w+', np.float16, (N, 4, Fd))
    for r in range(4): out[:, r] = V[r][:, 0]
    out.flush(); del out; os.replace(f'{B}/bank__L{l}.npy.part', f'{B}/bank__L{l}.npy')
I = [np.load(f'{D}/{RB}r{a}__idx.npy') for a in (0, 90, 180, 270)]; assert all((i == I[0]).all() for i in I)
np.save(f'{B}/bank__idx.npy', I[0])
for f in ['bank__L0.npy', 'bank__L1.npy', 'bank__L2.npy', 'bank__idx.npy']:
    if not os.path.lexists(f'{D}/{f}'): os.symlink(os.path.realpath(f'{B}/{f}'), f'{D}/{f}')
print('ASSEMBLE_ROTB_DONE', tag, s, flush=True)
