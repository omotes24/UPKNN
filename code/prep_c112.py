# prep_c112.py -- PROTOCOL_HIRES2.md: every 224-px image of PROTOCOL_HIRES.md (hc224_<set>.npy) -> 112 px by one 2x BOX reduction ->
#   256 px by one bicubic resize; saved as hc112_<set>.npy [N, 256, 256, 3] uint8 (same images and order as hc224_<set>).
import os
import numpy as np
from PIL import Image
from multiprocessing import Pool
ROOT = os.path.expanduser('~/upknn'); OUT = f'{ROOT}/data/npy'
SETS = ['imagenet_test', 'imagenet_bank', 'ssb_hard', 'ninco', 'inaturalist', 'textures', 'openimage_o', 'imagenet_v2', 'imagenet_c', 'imagenet_r']


def one(a):
    im = Image.fromarray(np.asarray(a)).resize((112, 112), Image.BOX)
    return np.asarray(im.resize((256, 256), Image.BICUBIC))


if __name__ == '__main__':
    for s in SETS:
        out = f'{OUT}/hc112_{s}.npy'
        if os.path.exists(out): print('exists', s, flush=True); continue
        X = np.load(f'{OUT}/hc224_{s}.npy', mmap_mode='r'); assert X.shape[1:] == (224, 224, 3), (s, X.shape)
        R = []
        with Pool(12) as pool:
            for i in range(0, len(X), 2000): R += pool.map(one, [np.asarray(x) for x in X[i:i + 2000]], chunksize=32)
        Y = np.stack(R); assert Y.shape == (len(X), 256, 256, 3) and Y.dtype == np.uint8
        np.save(out + '.tmp.npy', Y); os.replace(out + '.tmp.npy', out); print('[saved]', s, Y.shape, flush=True)
    print('PREP_C112_DONE', flush=True)
