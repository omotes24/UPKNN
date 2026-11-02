# prep_hr_cs.py -- the covariate-shifted ID sets of OpenOOD v1.5 (ImageNet-V2, -C, -R) at 256x256 for the ImageNet-256 tier
#   (PROTOCOL_IN256.md): exactly the 1,000 images per set of the ImageNet-64 benchmark's fixed subsample (runs/ood1k/s1k/in64_<set>.json,
#   indices into OpenOOD's lists test_imagenet_{v2,c,r}.txt), preprocessed as prep_hr.py (prep_newood.crop64 with size 256).
#   output: data/npy/hr_imagenet_{v2,c,r}.npy uint8 [1000, 256, 256, 3] (+ _paths.txt, _src_idx.npy)
import os, sys, json
import numpy as np
from PIL import Image
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_newood import crop64
HOME = os.path.expanduser('~'); ROOT = f'{HOME}/upknn'; OUT = f'{ROOT}/data/npy'
L = f'{ROOT}/data/openood/benchmark_imglist/imagenet'; LS = f'{HOME}/datasets/openood_official/images_largescale'
SRC = {'v2': f'{HOME}/datasets/extra_ood/imagenetv2-matched-frequency-format-val/', 'c': f'{LS}/imagenet_c/',
       'r': f'{HOME}/datasets/extra_ood/imagenet-r/'}


def one(path):
    with Image.open(path) as im: return crop64(im, 256)


if __name__ == '__main__':
    for k, root in SRC.items():
        name = f'hr_imagenet_{k}'
        if os.path.exists(f'{OUT}/{name}.npy'): print('exists', name, flush=True); continue
        rel = [l.rsplit(' ', 1)[0] for l in open(f'{L}/test_imagenet_{k}.txt').read().strip().split('\n')]
        sel = [int(i) for i in json.load(open(f'{ROOT}/runs/ood1k/s1k/in64_imagenet_{k}.json'))]
        paths = [root + rel[i].split('/', 1)[1] for i in sel]
        missing = [p for p in paths if not os.path.exists(p)]; assert not missing, missing[:3]
        with Pool(20) as pool: arr = np.stack(pool.map(one, paths, chunksize=16))
        np.save(f'{OUT}/{name}.npy', np.ascontiguousarray(arr, dtype=np.uint8)); np.save(f'{OUT}/{name}_src_idx.npy', np.asarray(sel, np.int64))
        open(f'{OUT}/{name}_paths.txt', 'w').write('\n'.join(paths) + '\n')
        print(f'[saved] {name} {arr.shape} mean={arr.mean():.2f}', flush=True)
    print('PREP_HR_CS_DONE', flush=True)
