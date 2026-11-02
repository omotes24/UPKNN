# prep_medmnist.py -- MedMNIST v2 (Yang et al., 2023) sets at 32x32 for two further in-distribution tiers of the ICML version:
#   ChestMNIST (frontal chest X-rays, NIH ChestX-ray14; grey) as an ID with a strong canonical orientation and PathMNIST (colon
#   histology patches, NCT-CRC-HE-100K; RGB) as an ID without one; OrganAMNIST, PneumoniaMNIST, BloodMNIST and DermaMNIST as
#   OOD sets of the two tiers.  The 64x64 release is resized to 32x32 (PIL bilinear, antialiased), uint8; grey sets are [N, 32, 32]
#   (g32_ prefix, like the FashionMNIST tier), RGB sets [N, 32, 32, 3].  Run: python code/prep_medmnist.py (from the repository root)
import os, numpy as np
from PIL import Image
import medmnist
ROOT = os.path.expanduser('~/upknn'); NPY = f'{ROOT}/data/npy'; RAW = f'{ROOT}/data/raw/medmnist'
os.makedirs(RAW, exist_ok=True)


def load(flag, split):
    cls = getattr(medmnist, medmnist.INFO[flag]['python_class'])
    return cls(split=split, download=True, root=RAW, size=64).imgs                      # uint8 [N, 64, 64(, 3)]


def to32(x):
    out = np.stack([np.asarray(Image.fromarray(im).resize((32, 32), Image.BILINEAR)) for im in x])
    return np.ascontiguousarray(out.astype(np.uint8))


def save(name, x):
    p = f'{NPY}/{name}.npy'
    if os.path.exists(p): print('exists', p, np.load(p, mmap_mode='r').shape); return
    np.save(p, x); print('saved', name, x.shape, x.dtype)


# ID tiers: train split for the diffusion model, test split for the evaluation (development / final images by the usual permutation)
save('g32_chest_train', to32(load('chestmnist', 'train')))
save('g32_chest_test', to32(load('chestmnist', 'test')))
save('path_train', to32(load('pathmnist', 'train')))
save('path_test', to32(load('pathmnist', 'test')))
# OOD sets: all splits concatenated where the test split is small
save('g32_organa_test', to32(load('organamnist', 'test')))
save('g32_pneumonia_all', to32(np.concatenate([load('pneumoniamnist', s) for s in ('train', 'val', 'test')])))
save('blood_all', to32(np.concatenate([load('bloodmnist', s) for s in ('train', 'val', 'test')])))
save('derma_all', to32(np.concatenate([load('dermamnist', s) for s in ('train', 'val', 'test')])))
# cross versions: the grey histology patches for the chest tier, the X-rays as RGB for the histology tier
path_test = np.load(f'{NPY}/path_test.npy'); chest_test = np.load(f'{NPY}/g32_chest_test.npy')
save('g32_path_gray_test', np.ascontiguousarray(np.stack([np.asarray(Image.fromarray(im).convert('L')) for im in path_test]).astype(np.uint8)))
save('chest_rgb_test', np.ascontiguousarray(np.repeat(chest_test[..., None], 3, axis=3)))
print('PREP_DONE')
