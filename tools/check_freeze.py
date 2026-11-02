# check_freeze.py -- verifies that the frozen files of PROTOCOL_UKNN.md in this repository are the ones whose checksums were recorded
#   when the protocol was frozen (protocols/FREEZE_UKNN.md5, written on the GPU server before the final evaluation): it undoes the
#   substitutions of tools/subs.py and compares the MD5 checksums.   usage (repository root): python tools/check_freeze.py
import os, sys, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from subs import INVERTIBLE
MAP = {'runs/kx/PROTOCOL_UKNN.md': 'protocols/PROTOCOL_UKNN.md'}
ok = True
for line in open('protocols/FREEZE_UKNN.md5'):
    parts = line.split()
    if len(parts) != 2 or len(parts[0]) != 32: continue
    md5, path = parts; f = MAP.get(path, path)
    t = open(f, encoding='utf-8').read()
    for a, b in INVERTIBLE[::-1]: t = t.replace(b, a) if b.strip() else t
    got = hashlib.md5(t.encode('utf-8')).hexdigest(); ok &= got == md5
    print(('OK  ' if got == md5 else 'DIFF'), f, got)
sys.exit(0 if ok else 1)
