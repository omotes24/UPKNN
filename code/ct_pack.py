# ct_pack.py -- merges the per-scale outputs of ct_extract.py into one file per ID model.
#   python code/ct_pack.py sel   -> runs/ct/pack_<tag>.npz      (ID development images and extra OOD sets only: the selection data)
#   python code/ct_pack.py fin   -> runs/ct/packfin_<tag>.npz   (the main-evaluation images; written only after the choice is frozen)
import os, sys, json
import numpy as np
ROOT = os.path.expanduser('~/upknn'); CT = f'{ROOT}/runs/ct'
which = sys.argv[1]; assert which in ('sel', 'fin')
for tag in ['edmc10', 'l6svhn', 'dpceleba', 'edmc100', 'in64', 'fmnist']:
    sets = json.load(open(f'{CT}/sets_{tag}.json'))
    names = [r[0] for r in sets if (r[0].startswith('fin-') if which == 'fin' else not r[0].startswith('fin-'))]
    if not all(os.path.exists(f'{CT}/{tag}/k{k}/DONE') for k in range(10)): print(tag, 'not complete'); continue
    out = {}
    for n in names:
        zs = [np.load(f'{CT}/{tag}/k{k}/{n}.npz') for k in range(10)]
        assert all(np.array_equal(z['idx'], zs[0]['idx']) for z in zs)
        out['K__' + n] = np.stack([z['K'] for z in zs], 1).astype(np.float32); out['idx__' + n] = zs[0]['idx']
    fn = f'{CT}/{"pack" if which == "sel" else "packfin"}_{tag}.npz'
    np.savez(fn, **out)
    print(tag, which, len(names), 'sets ->', fn, f'{os.path.getsize(fn) / 1e6:.0f} MB')
