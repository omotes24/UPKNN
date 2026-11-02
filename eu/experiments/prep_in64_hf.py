# prep_in64_hf.py -- ImageNet-1k training images at 64x64 for the ImageNet-64 rotation classifiers (the 10k-image array of the
#   project is too small to train a ResNet-18).  Source: the first train shard of the Hugging Face dataset
#   benjamin-paine/imagenet-1k-64x64 (center crop to the short side, Lanczos resize to 64x64; ImageNet train split, disjoint from the
#   OpenOOD ImageNet val/test images of the benchmark).  -> data/npy/in64_imagenet_train_hf.npy (uint8 [N, 64, 64, 3])
import os, io
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from multiprocessing import Pool
ROOT = os.environ.get('SR_ROOT', os.path.expanduser('~/upknn'))
OUT = f'{ROOT}/data/npy/in64_imagenet_train_hf.npy'


def dec(b):
    im = Image.open(io.BytesIO(b)).convert('RGB')
    assert im.size == (64, 64), im.size
    return np.asarray(im, dtype=np.uint8)


if __name__ == '__main__':
    p = hf_hub_download('benjamin-paine/imagenet-1k-64x64', 'data/train-00000-of-00005.parquet', repo_type='dataset')
    t = pq.read_table(p)
    print(t.schema, t.num_rows, flush=True)
    col = t.column('image').to_pylist()
    bs = [c['bytes'] for c in col]
    with Pool(12) as pool: arr = np.stack(pool.map(dec, bs, chunksize=1024))
    np.save(OUT, arr)
    print('saved', OUT, arr.shape, arr.mean(), flush=True)
