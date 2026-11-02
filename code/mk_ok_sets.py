# mk_ok_sets.py -- image lists of PROTOCOL_OK.md: selection data = all ID development images of the development run (DEVRUN, the
#   images of the rule selection) + the extra OOD sets of PROTOCOL_CT (runs/ct/sets_<tag>.json, 'sel-*'); held-out evaluation = the
#   images of the held-out split (runs/ct_fresh/sets_<tag>.json, 'fresh-*', the images of every method's held-out runs).
#   usage: python code/mk_ok_sets.py  -> runs/ok/sets_<tag>.json (selection + held-out), [[name, data, idx, pre], ...]
import os, json
import numpy as np
ROOT = os.path.expanduser('~/upknn')
DEVRUN = {'edmc10': 'B_edmc10', 'l6svhn': 'P_l6svhn', 'dpceleba': 'P_dpceleba', 'edmc100': 'P_edmc100', 'in64': 'I2_edm2xs', 'fmnist': 'P_fmnist'}
IDN = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'edmc100': 'cifar100_test', 'in64': 'in64_imagenet_test',
       'fmnist': 'g32_fmnist_test'}
os.makedirs(f'{ROOT}/runs/ok', exist_ok=True)
for t, idn in IDN.items():
    dv = sorted(int(i) for i in np.load(f'{ROOT}/runs/feat/{DEVRUN[t]}/{idn}.npz', allow_pickle=True)['idx'].tolist())
    rows = [['devall-' + idn, idn, dv, 'none']]
    ct = json.load(open(f'{ROOT}/runs/ct/sets_{t}.json'))
    cdev = [r for r in ct if r[0].startswith('dev-')][0]
    assert set(cdev[2]) <= set(dv), t
    rows += [r for r in ct if r[0].startswith('sel-')]
    fr = json.load(open(f'{ROOT}/runs/ct_fresh/sets_{t}.json'))
    rows += [r for r in fr if r[0].startswith('fresh-')]
    assert len({r[0] for r in rows}) == len(rows)
    json.dump(rows, open(f'{ROOT}/runs/ok/sets_{t}.json', 'w'))
    print(t, len(rows), 'sets', sum(len(r[2]) for r in rows), 'images;', [(r[0], len(r[2]), r[3]) for r in rows if r[3] != 'none'])
