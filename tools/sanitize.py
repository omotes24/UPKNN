# sanitize.py -- applies tools/subs.py to the files copied from the GPU server (run once, from the repository root).
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subs import SUBS
FILES = sorted(glob.glob('code/*.py') + glob.glob('code/*.sh') + glob.glob('code/jobs/*.txt') + glob.glob('eu/experiments/*.py') + glob.glob('protocols/*.md'))
n = 0
for f in FILES:
    t = open(f, encoding='utf-8').read(); t0 = t
    for a, b in SUBS:
        assert b == '' or b not in t0 or a in t0, f'{f}: replacement {b!r} already present'
        t = t.replace(a, b)
    if t != t0: open(f, 'w', encoding='utf-8').write(t); n += 1
print('sanitized', n, 'of', len(FILES), 'files')
