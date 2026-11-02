# prep_hr.py -- 256x256 RGB arrays for Experiment A (ADM ImageNet-256, pixel space), ADM / EDM2 preprocessing (prep_newood.crop64 with
#   size 256: BOX halvings while the shorter side is >= 512, bicubic to 256 on the shorter side, centre crop).  Sources already on GPU-server
#   (OpenOOD ImageNet benchmark lists and images, as for the ImageNet-64 tier; Places365 val_256; CelebA-HQ 256 from the Hugging Face
#   hub, one shard).  Arrays (data/npy/hr_<set>.npy, uint8 [N,256,256,3]):
#     hr_imagenet_test  8000 of the 45,000 OpenOOD ImageNet test images (fixed random subset; ID: first 2000 of the Dragon Glider permutation
#                       = development / calibration images, the rest = final split)
#     hr_imagenet_bank  10,000 ImageNet training images (10 per class, full resolution, same crop) for the Cu and Cnn references
#     hr_ssb_hard, hr_ninco, hr_inaturalist, hr_openimage_o, hr_textures, hr_places365, hr_celebahq   2000 each
#     hr_synth_noise, hr_synth_const  1000 each
#   usage: python code/prep_hr.py [set ...]
import os, sys, glob, io, traceback
import numpy as np
from PIL import Image
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_newood import crop64
HOME = os.path.expanduser('~'); ROOT = f'{HOME}/upknn/data'; OUT = f'{ROOT}/npy'
L = f'{ROOT}/openood/benchmark_imglist/imagenet'; D = f'{HOME}/datasets/openood_official'
N_OOD, N_ID, N_BANK = 2000, 8000, 10000


def one(path):
    try:
        with Image.open(path) as im: return crop64(im, 256)
    except Exception as e:
        print('ERR', path, e, flush=True); return np.zeros((256, 256, 3), np.uint8)


def read_list(f):
    rows = [l.rsplit(' ', 1) for l in open(f'{L}/{f}').read().strip().split('\n')]
    return [r[0] for r in rows]


def pick(paths, n, seed=0):
    rng = np.random.RandomState(seed); return [paths[i] for i in sorted(rng.choice(len(paths), min(n, len(paths)), replace=False))]


def build(name, paths, fn=one):
    if os.path.exists(f'{OUT}/{name}.npy'): print('exists', name, flush=True); return
    missing = [p for p in paths if not os.path.exists(p)]
    print(name, len(paths), 'missing', len(missing), missing[:2], flush=True); assert not missing
    with Pool(20) as pool: arr = np.stack(pool.map(fn, paths, chunksize=32))
    np.save(f'{OUT}/{name}.npy', np.ascontiguousarray(arr, dtype=np.uint8)); open(f'{OUT}/{name}_paths.txt', 'w').write('\n'.join(paths) + '\n')
    print(f'[saved] {name} {arr.shape} mean={arr.mean():.2f} std={arr.std():.2f}', flush=True)


def imagenet_test():
    rel = read_list('test_imagenet.txt'); build('hr_imagenet_test', pick([f'{D}/images_largescale/imagenet_1k/val/{os.path.basename(r)}' for r in rel], N_ID))


def imagenet_bank():
    root = f'{D}/images_largescale/imagenet_1k/train'; paths = []
    for c in sorted(d for d in os.listdir(root) if d.startswith('n')):
        paths += sorted(glob.glob(f'{root}/{c}/*'))[:N_BANK // 1000]
    build('hr_imagenet_bank', paths)


def ssb_hard(): build('hr_ssb_hard', pick([f'{HOME}/ood_benchmarks/' + r for r in read_list('test_ssb_hard.txt')], N_OOD))


def ninco():
    rel = read_list('test_ninco.txt'); nroot = f'{HOME}/ood_benchmarks/ninco/NINCO/NINCO_OOD_classes'
    nidx = {os.path.relpath(q, nroot).replace(' ', '_'): q for q in glob.glob(f'{nroot}/*/*')}
    build('hr_ninco', pick([nidx.get(r.split('/', 1)[1], f'{nroot}/' + r.split('/', 1)[1]) for r in rel], N_OOD))


def inaturalist(): build('hr_inaturalist', pick([f'{D}/images_largescale/inaturalist/extracted/' + r.split('/', 1)[1] for r in read_list('test_inaturalist.txt')], N_OOD))
def openimage_o(): build('hr_openimage_o', pick([f'{D}/images_largescale/openimage_o/extracted/' + r.split('/', 1)[1] for r in read_list('test_openimage_o.txt')], N_OOD))
def textures(): build('hr_textures', pick([f'{D}/images_classic/' + r for r in read_list('test_textures.txt')], N_OOD))
def places365(): build('hr_places365', pick(sorted(glob.glob(f'{HOME}/datasets/places365_val256/val_256/*.jpg')), N_OOD))


def celebahq():
    if os.path.exists(f'{OUT}/hr_celebahq.npy'): print('exists hr_celebahq'); return
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    f = hf_hub_download('korexyz/celeba-hq-256x256', 'data/train-00000-of-00006.parquet', repo_type='dataset', cache_dir=f'{ROOT}/raw/hf')
    t = pq.read_table(f); col = [c for c in t.column_names if 'image' in c.lower()][0]; rows = t.column(col).to_pylist()
    rng = np.random.RandomState(0); sel = sorted(rng.choice(len(rows), min(N_OOD, len(rows)), replace=False))
    arr = np.stack([crop64(Image.open(io.BytesIO(rows[i]['bytes'])), 256) for i in sel])
    np.save(f'{OUT}/hr_celebahq.npy', np.ascontiguousarray(arr, dtype=np.uint8)); print(f'[saved] hr_celebahq {arr.shape} mean={arr.mean():.2f} std={arr.std():.2f}', flush=True)


def synth():
    rng = np.random.RandomState(0)
    if not os.path.exists(f'{OUT}/hr_synth_noise.npy'): np.save(f'{OUT}/hr_synth_noise.npy', rng.randint(0, 256, (1000, 256, 256, 3), dtype=np.uint8)); print('[saved] hr_synth_noise')
    if not os.path.exists(f'{OUT}/hr_synth_const.npy'):
        c = rng.randint(0, 256, (1000, 1, 1, 3), dtype=np.uint8); np.save(f'{OUT}/hr_synth_const.npy', np.ascontiguousarray(np.broadcast_to(c, (1000, 256, 256, 3)))); print('[saved] hr_synth_const')


if __name__ == '__main__':
    todo = sys.argv[1:] or ['imagenet_test', 'imagenet_bank', 'ssb_hard', 'ninco', 'inaturalist', 'openimage_o', 'textures', 'places365', 'celebahq', 'synth']
    for t in todo:
        try: print('==', t, flush=True); globals()[t]()
        except Exception: traceback.print_exc(); print('FAILED', t, flush=True)
    print('PREP_HR_DONE', flush=True)
