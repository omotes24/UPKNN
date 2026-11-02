# prep_ood1k.py -- the OpenOOD v1.5 ImageNet-1K benchmark at 64x64 for the frozen ImageNet-64 detectors:
#   (1) the covariate-shifted ID sets of OpenOOD v1.5 (full-spectrum benchmark) as 64x64 arrays, with the SAME preprocessing as
#       prep_in64.py (crop64: box halvings while the short side >= 128, bicubic to 64 on the short side, centre crop):
#       ImageNet-V2 (matched frequency, 10,000), ImageNet-C (the OpenOOD sample of 10,000 over 15 corruptions x 5 severities),
#       ImageNet-R (30,000), in the order of OpenOOD's lists benchmark_imglist/imagenet/test_imagenet_{v2,c,r}.txt;
#   (2) the benchmark sets (prefix ood1k-) appended to runs/ok/sets_in64.json (a backup is kept): the OpenOOD test lists of the ID set
#       (ImageNet val, 45,000), near-OOD (SSB-hard, NINCO) and far-OOD (iNaturalist, Textures, OpenImage-O) WITHOUT the images used to
#       develop or select Upright kNN / Vertical Limit or in their earlier final evaluation (devall-, sel-, fresh-, final2- of the same
#       file), and the three covariate-shift sets in full.
#   usage: python code/prep_ood1k.py
import os, sys, json, zipfile, shutil, time
import numpy as np
from PIL import Image
from multiprocessing import Pool
HOME = os.path.expanduser('~'); ROOT = f'{HOME}/upknn'; OUT = f'{ROOT}/data/npy'
L = f'{ROOT}/data/openood/benchmark_imglist/imagenet'; LS = f'{HOME}/datasets/openood_official/images_largescale'


def crop64(im, size=64):                                              # verbatim from prep_in64.py
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


def build(name, paths):
    if os.path.exists(f'{OUT}/{name}.npy'): print('exists', name, flush=True); return
    missing = [p for p in paths if not os.path.exists(p)]
    print(name, len(paths), 'missing', len(missing), missing[:2], flush=True); assert not missing
    with Pool(22) as pool: arr = np.stack(pool.map(_one, paths, chunksize=64))
    np.save(f'{OUT}/{name}.npy', arr); print(f'[saved] {name} {arr.shape} mean={arr.mean():.2f} std={arr.std():.2f}', flush=True)


def read_list(f):
    rows = [l.rsplit(' ', 1) for l in open(f'{L}/{f}').read().strip().split('\n')]
    return [r[0] for r in rows], np.array([int(r[1]) for r in rows])


t0 = time.time()
if not os.path.isdir(f'{LS}/imagenet_c'):
    with zipfile.ZipFile(f'{HOME}/datasets/openood_official/dl/imagenet_c.zip') as z: z.extractall(f'{LS}/imagenet_c')
rel, y = read_list('test_imagenet_v2.txt')
build('in64_imagenet_v2', [f'{HOME}/datasets/extra_ood/imagenetv2-matched-frequency-format-val/' + r.split('/', 1)[1] for r in rel]); np.save(f'{OUT}/in64_imagenet_v2_y.npy', y)
rel, y = read_list('test_imagenet_c.txt')
build('in64_imagenet_c', [f'{LS}/imagenet_c/' + r.split('/', 1)[1] for r in rel]); np.save(f'{OUT}/in64_imagenet_c_y.npy', y)
json.dump([r.split('/')[1:3] for r in rel], open(f'{OUT}/in64_imagenet_c_meta.json', 'w'))   # [corruption, severity] per image
rel, y = read_list('test_imagenet_r.txt')
build('in64_imagenet_r', [f'{HOME}/datasets/extra_ood/imagenet-r/' + r.split('/', 1)[1] for r in rel]); np.save(f'{OUT}/in64_imagenet_r_y.npy', y)
# ---- the benchmark sets
SF = f'{ROOT}/runs/ok/sets_in64.json'; S = json.load(open(SF))
if any(r[0].startswith('ood1k-') for r in S): print('ood1k sets exist already'); sys.exit(0)
shutil.copy(SF, f'{ROOT}/runs/ok/sets_in64.before_ood1k.json')
used = {}
for name, data, idx, pre in S:
    used.setdefault(data, set()).update(int(i) for i in idx)
new = []
for data in ['in64_imagenet_test', 'in64_ssb_hard', 'in64_ninco', 'in64_inaturalist', 'in64_textures', 'in64_openimage_o']:
    n = len(np.load(f'{OUT}/{data}.npy', mmap_mode='r')); keep = [i for i in range(n) if i not in used.get(data, set())]
    new.append([f'ood1k-{data}', data, keep, 'none']); print(f'ood1k-{data}: {len(keep)} of {n} (excluded {n - len(keep)})', flush=True)
for data in ['in64_imagenet_v2', 'in64_imagenet_c', 'in64_imagenet_r']:
    n = len(np.load(f'{OUT}/{data}.npy', mmap_mode='r')); new.append([f'ood1k-{data}', data, list(range(n)), 'none']); print(f'ood1k-{data}: {n}', flush=True)
json.dump(S + new, open(SF, 'w'))
print('PREP_OOD1K_DONE', f'{time.time() - t0:.0f}s', flush=True)
