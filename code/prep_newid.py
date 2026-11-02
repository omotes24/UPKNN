# prep_newid.py -- training / test arrays for two NEW in-distribution tiers (own EDM-type models, train.py), chosen after the rule was frozen:
#   tin_train      Tiny-ImageNet train (100,000 images, 200 classes) at 32x32 (prep_data.resize_cc; the ID test set is tin_val = the
#                  official validation images, already prepared);  no development selection is made on these tiers
#   eurosat_train / eurosat_test   EuroSAT RGB (27,000 patches): fixed permutation (seed 0), first 5,000 = test, the rest = train
#   Appends the two training jobs to runs/jobs/NEWX.txt as soon as each array exists (append-only queue, consumed by dg_X* workers).
import os, sys, glob, time, zipfile
import numpy as np
from PIL import Image
from multiprocessing import Pool
ROOT = os.path.expanduser('~/upknn'); RAW, OUT = f'{ROOT}/data/raw', f'{ROOT}/data/npy'
os.chdir(ROOT)


def resize_cc(img, size=32):
    img = img.convert('RGB'); w, h = img.size; s = size / min(w, h)
    nw, nh = max(size, round(w * s)), max(size, round(h * s))
    if (nw, nh) != (w, h): img = img.resize((nw, nh), Image.BILINEAR)
    l, t = (nw - size) // 2, (nh - size) // 2
    return np.asarray(img.crop((l, t, l + size, t + size)))


def _load(path):
    try:
        with Image.open(path) as im: return resize_cc(im)
    except Exception as e:
        print('ERR', path, e, flush=True); return np.zeros((32, 32, 3), np.uint8)


def save(name, arr):
    arr = np.ascontiguousarray(arr, dtype=np.uint8); assert arr.shape[1:] == (32, 32, 3), (name, arr.shape)
    np.save(f'{OUT}/{name}.npy', arr); print(f'[saved] {name} {arr.shape} mean={arr.mean():.2f} std={arr.std():.2f}', flush=True)


def append_job(line):
    if line in open('runs/jobs/NEWX.txt').read().split('\n'): print('already queued:', line[:80], flush=True); return
    open('runs/jobs/NEWX.txt', 'a').write(line + '\n'); print('appended:', line[:120], flush=True)


# ---------------------------------------------------------------- Tiny-ImageNet train, 32x32
if not os.path.exists(f'{OUT}/tin_train.npy'):
    z = f'{RAW}/tiny-imagenet-200.zip'
    if not os.path.exists(f'{RAW}/tiny-imagenet-200/train'):
        with zipfile.ZipFile(z) as zf: zf.extractall(RAW)
    paths = sorted(glob.glob(f'{RAW}/tiny-imagenet-200/train/*/images/*.JPEG'))
    print('tin train images', len(paths), flush=True); assert len(paths) == 100000
    with Pool(20) as p: arr = np.stack(p.map(_load, paths, chunksize=256))
    save('tin_train', arr)
    wn = sorted({os.path.basename(os.path.dirname(os.path.dirname(q))) for q in paths}); lab = {w: i for i, w in enumerate(wn)}
    np.save(f'{OUT}/tin_train_y.npy', np.array([lab[os.path.basename(os.path.dirname(os.path.dirname(q)))] for q in paths]))
append_job('python code/train.py --gpu {gpu} --data tin_train --out models/tin_m --arch m --bs 128 --lr 2.5e-4 --warm 0.2 --mimg 8 --calib_n 5000 >> logs/train_tin_m.log 2>&1')

# ---------------------------------------------------------------- EuroSAT split (waits for prep_newood.py's eurosat.npy)
t0 = time.time()
while not os.path.exists(f'{OUT}/eurosat.npy'):
    if time.time() - t0 > 7200: print('eurosat.npy did not appear; no EuroSAT tier', flush=True); sys.exit(0)
    if os.path.exists('logs/prep_newood.log') and 'FAILED eurosat' in open('logs/prep_newood.log', errors='ignore').read():
        print('EuroSAT preparation failed; no EuroSAT tier', flush=True); sys.exit(0)
    time.sleep(30)
time.sleep(10)
if not os.path.exists(f'{OUT}/eurosat_test.npy'):
    x = np.load(f'{OUT}/eurosat.npy'); perm = np.random.RandomState(0).permutation(len(x))
    save('eurosat_test', x[np.sort(perm[:5000])]); save('eurosat_train', x[np.sort(perm[5000:])])
append_job('python code/train.py --gpu {gpu} --data eurosat_train --out models/eurosat_m --arch m --bs 128 --lr 2.5e-4 --warm 0.2 --mimg 4 --calib_n 2000 >> logs/train_eurosat_m.log 2>&1')
print('PREP_NEWID_DONE', flush=True)
