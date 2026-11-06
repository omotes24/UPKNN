# mk_e5b.py -- PROTOCOL_E5B.md: nested ImageNet-64 banks of 25 / 50 training images per class, preprocessed exactly as the
#   benchmark's images (crop64 of code/prep_in64.py), and the set lists of the bank tiers.  CPU only.
import os, sys, json
import numpy as np
from PIL import Image
from multiprocessing import Pool
HOME = os.path.expanduser('~'); ROOT = f'{HOME}/upknn'; NPY = f'{ROOT}/data/npy'
D = f'{HOME}/datasets/openood_official/images_largescale/imagenet_1k/train'


def crop64(im, size=64):                                              # copy of prep_in64.crop64 ("center-crop-dhariwal")
    im = im.convert('RGB')
    while min(*im.size) >= 2 * size:
        im = im.resize(tuple(x // 2 for x in im.size), resample=Image.BOX)
    s = size / min(*im.size)
    im = im.resize(tuple(round(x * s) for x in im.size), resample=Image.BICUBIC)
    a = np.asarray(im)
    y, x = (a.shape[0] - size) // 2, (a.shape[1] - size) // 2
    return a[y:y + size, x:x + size]


def _one(path):
    with Image.open(path) as im:
        return crop64(im)


if __name__ == '__main__':
    wn = sorted(w for w in os.listdir(D) if os.path.isdir(f'{D}/{w}'))
    assert len(wn) == 1000, len(wn)
    per = {w: sorted(os.listdir(f'{D}/{w}')) for w in wn}
    assert all(len(v) >= 50 for v in per.values())
    S = [r for r in json.load(open(f'{ROOT}/runs/ok/sets_in64.json')) if r[0].startswith('ood1k-')]
    ref = np.load(f'{NPY}/in64_imagenet_train10.npy', mmap_mode='r')
    for k in (25, 50):
        out = f'{NPY}/in64_bank_{k}k.npy'
        if not os.path.exists(out):
            paths = [f'{D}/{w}/{f}' for w in wn for f in per[w][:k]]
            with Pool(12) as pool: arr = np.stack(pool.map(_one, paths, chunksize=256))
            assert arr.shape == (1000 * k, 64, 64, 3) and arr.dtype == np.uint8, arr.shape
            # nesting check: the first 10 per class equal the protocol bank (same files, same preprocessing)
            first10 = arr.reshape(1000, k, 64, 64, 3)[:, :10].reshape(-1, 64, 64, 3)
            assert first10.shape == ref.shape and np.array_equal(first10, ref), 'bank not nested with in64_imagenet_train10'
            np.save(out + '.tmp.npy', np.ascontiguousarray(arr)); os.replace(out + '.tmp.npy', out)
        p = f'{ROOT}/runs/ok/sets_in64b{k}.json'
        if not os.path.exists(p): json.dump(S, open(p, 'w'))
        print('bank', k, 'k', np.load(out, mmap_mode='r').shape, 'sets', len(S), flush=True)
    print('E5B_MK_DONE', flush=True)
