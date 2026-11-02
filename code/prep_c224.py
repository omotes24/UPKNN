# prep_c224.py -- resolution-controlled ImageNet sets (PROTOCOL_HIRES.md): every image -> 224 px (prep_newood.crop64, size 224), saved as
#   hc224_<set>.npy [N, 224, 224, 3] and, bicubic 224 -> 256, hc_<set>.npy [N, 256, 256, 3] (uint8); same images and order as hr_<set>.
#   usage: python code/prep_c224.py      (the queue line then writes the set file and touches logs/hires/PREP_DONE)
import os, sys
import numpy as np
from PIL import Image
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_newood import crop64
ROOT = os.path.expanduser('~/upknn'); OUT = f'{ROOT}/data/npy'
SETS = ['imagenet_test', 'imagenet_bank', 'ssb_hard', 'ninco', 'inaturalist', 'textures', 'openimage_o', 'imagenet_v2', 'imagenet_c', 'imagenet_r']


def one(path):
    with Image.open(path) as im: a = crop64(im, 224)
    return a, np.asarray(Image.fromarray(a).resize((256, 256), Image.BICUBIC))


if __name__ == '__main__':
    for s in SETS:
        if os.path.exists(f'{OUT}/hc_{s}.npy'): print('exists', s, flush=True); continue
        P = [l for l in open(f'{OUT}/hr_{s}_paths.txt').read().split('\n') if l]
        n = len(np.load(f'{OUT}/hr_{s}.npy', mmap_mode='r')); assert len(P) == n, (s, len(P), n)
        with Pool(20) as pool: R = pool.map(one, P, chunksize=16)
        np.save(f'{OUT}/hc224_{s}.npy', np.stack([r[0] for r in R])); np.save(f'{OUT}/hc_{s}.npy', np.stack([r[1] for r in R]))
        print('[saved]', s, len(R), flush=True)
    print('PREP_C224_DONE', flush=True)
