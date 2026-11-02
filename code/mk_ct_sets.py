# mk_ct_sets.py -- image lists of PROTOCOL_CT.md: selection data (the ID development images and the extra OOD sets, images
#   common to the pair and +eps-only runs) and the main evaluation (the common images of the 63 pairs, runs/ct/common_idx.json)
#   usage: python code/mk_ct_sets.py  -> runs/ct/sets_<tag>.json   [[name, data, idx, pre], ...]
import os, json
import numpy as np
ROOT = os.path.expanduser('~/upknn')
RGB = ['cifar10_test', 'cifar100_test', 'svhn_test', 'celeba_test', 'textures', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val']
TIERS = {
    'edmc10':   dict(id='cifar10_test', ood=[s for s in RGB if s != 'cifar10_test'] + ['cifar10_test__vflip', 'synth_noise', 'synth_const']),
    'l6svhn':   dict(id='svhn_test', ood=[s for s in RGB if s != 'svhn_test'] + ['svhn_test__vflip', 'synth_noise', 'synth_const']),
    'dpceleba': dict(id='celeba_test', ood=[s for s in RGB if s != 'celeba_test'] + ['celeba_test__vflip', 'synth_noise', 'synth_const']),
    'edmc100':  dict(id='cifar100_test', ood=[s for s in RGB if s != 'cifar100_test'] + ['cifar100_test__vflip', 'synth_noise', 'synth_const']),
    'in64':     dict(id='in64_imagenet_test', ood=['in64_ssb_hard', 'in64_ninco', 'in64_inaturalist', 'in64_textures', 'in64_openimage_o', 'in64_places365', 'in64_celeba_test', 'in64_imagenet_test__vflip']),
    'fmnist':   dict(id='g32_fmnist_test', ood=['g32_mnist_test', 'g32_kmnist_test', 'g32_omniglot_eval', 'g32_notmnist', 'g32_fmnist_test__vflip', 'g32_synth_noise', 'g32_synth_const']),
}
RGBPOOL = ['stl10_test', 'gtsrb_test', 'flowers102_test', 'eurosat', 'mnist_rgb32', 'pets_test', 'caltech101', 'food101_test', 'aircraft_test', 'imagenette_val', 'sst2_test']
POOL = {'edmc10': RGBPOOL, 'l6svhn': RGBPOOL, 'dpceleba': RGBPOOL, 'edmc100': RGBPOOL,
        'in64': ['in64_' + s for s in RGBPOOL if s not in ('imagenette_val', 'mnist_rgb32')] + ['in64_mnist_rgb'],
        'fmnist': ['g32_emnist_letters', 'g32_k49', 'g32_semeion', 'g32_cifar10_gray', 'g32_svhn_gray', 'g32_gtsrb_gray']}
PAIR = {'edmc10': 'F_edmc10', 'l6svhn': 'F_l6svhn', 'dpceleba': 'F_dpceleba', 'edmc100': 'F_edmc100', 'in64': 'F2_in64', 'fmnist': 'F_fmnist'}
DEVRUN = {'edmc10': 'B_edmc10', 'l6svhn': 'P_l6svhn', 'dpceleba': 'P_dpceleba', 'edmc100': 'P_edmc100', 'in64': 'I2_edm2xs', 'fmnist': 'P_fmnist'}
COM = json.load(open(f'{ROOT}/runs/ct/common_idx.json'))


def idx(run, name): return set(np.load(f'{ROOT}/runs/feat/{run}/{name}.npz', allow_pickle=True)['idx'].tolist())


for t, T in TIERS.items():
    rows = [['dev-' + T['id'], T['id'], sorted(int(i) for i in idx(DEVRUN[t], T['id']) & idx(f'N_{t}', T['id'] + '__dev')), 'none']]
    for s in POOL[t]: rows.append(['sel-' + s, s, sorted(int(i) for i in idx(PAIR[t], s) & idx(f'N_{t}', s)), 'none'])
    for s in ['__id__'] + T['ood']:
        name = T['id'] if s == '__id__' else s
        rows.append(['fin-' + name, name.split('__')[0], sorted(int(i) for i in COM[t][s]), 'vflip' if name.endswith('__vflip') else 'none'])
    json.dump(rows, open(f'{ROOT}/runs/ct/sets_{t}.json', 'w'))
    print(t, len(rows), 'sets', sum(len(r[2]) for r in rows), 'images')
