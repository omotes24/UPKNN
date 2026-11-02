# ok_export_main.py -- the files the main evaluation of PROTOCOL_OK.md reads (ID development and held-out images, seeds 0-2): sims at
#   sigma_1 (full), errors at sigma_7 (similarity array dropped), the L012 neighbour indices, the similarity profiles and the bank
#   errors; written to ~/dg_ok_main/ with the cloud layout (sims/, nn/, prof/, s<seed>/<tag>/k<k>/bank__e.npy) and tarred.
import os, json, shutil
import numpy as np
R = os.path.expanduser('~/upknn/runs/ok'); O = os.path.expanduser('~/dg_ok_main')
TAGS = ['edmc10', 'l6svhn', 'dpceleba', 'edmc100', 'in64', 'fmnist']
n = 0
for seed in (0, 1, 2):
    for t in TAGS:
        names = [r[0] for r in json.load(open(f'{R}/sets_{t}.json')) if r[0].startswith(('devall-', 'fresh-'))]
        for k in (0, 6):
            os.makedirs(f'{O}/sims/s{seed}/{t}/k{k}', exist_ok=True); os.makedirs(f'{O}/s{seed}/{t}/k{k}', exist_ok=True)
            shutil.copy(f'{R}/s{seed}/{t}/k{k}/bank__e.npy', f'{O}/s{seed}/{t}/k{k}/bank__e.npy')
        os.makedirs(f'{O}/nn/s{seed}', exist_ok=True); os.makedirs(f'{O}/prof/s{seed}', exist_ok=True)
        for nm in names:
            shutil.copy(f'{R}/sims/s{seed}/{t}/k0/{nm}.npz', f'{O}/sims/s{seed}/{t}/k0/{nm}.npz')
            z = np.load(f'{R}/sims/s{seed}/{t}/k6/{nm}.npz')
            np.savez(f'{O}/sims/s{seed}/{t}/k6/{nm}.npz', S=np.zeros((len(z['idx']), 1, 1, 1, 1, 1), np.float16), e=z['e'], idx=z['idx'])
            z = np.load(f'{R}/nn/s{seed}/{t}__{nm}.npz'); np.savez(f'{O}/nn/s{seed}/{t}__{nm}.npz', i_L012=z['i_L012'][:, :50])
            shutil.copy(f'{R}/prof/s{seed}/{t}__{nm}.npz', f'{O}/prof/s{seed}/{t}__{nm}.npz'); n += 1
print('exported', n, 'sets')
