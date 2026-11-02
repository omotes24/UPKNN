# fig1_examples.py -- examples for Figure 1 of the Upright-kNN resume (CIFAR-10 as ID, development split fresh, noise seed 0; CPU only)
#   rule (no hand picking): the ID image whose score s(x) is the median of the ID images; the same image upside down; the Tiny-ImageNet image
#   whose score is the median of that set.  For each: the image, its 3 nearest reference images (mean of the three L2-normalised cosines,
#   the space of the content term), the content term, the orientation posterior q(r | x) of the MLP (softmax(logits / T)), the score.
#   output: runs/kx/fig1_examples.npz (images uint8) + runs/kx/fig1_examples.json
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.expanduser('~/upknn/code'))
from dgcore import NPY
from ct_extract_defs import bank_indices
R = os.path.expanduser('~/upknn'); tag = 'edmc10'
Z = np.load(f'{R}/runs/kx/export/{tag}_s0_fresh.npz')
def term(key, n): return Z[f'{key}__fresh-{n}'].astype(np.float64), Z[f'idx__fresh-{n}']
def feats(n):
    F = []
    for l in range(3):
        X = np.load(f'{R}/runs/ok/s0/{tag}/k0/{n}__L{l}.npy', mmap_mode='r'); C = X.shape[2] // 10
        v = np.asarray(X[:, 0, :2 * C], np.float32); F.append(v / np.linalg.norm(v, axis=1, keepdims=True))
    return F
B = feats('bank'); bank_img = np.load(f'{NPY}/cifar10_train.npy', mmap_mode='r'); bidx, _ = bank_indices(len(bank_img))
T = json.load(open(f'{R}/runs/kx/rothead/s0/{tag}_k0_gqR/temp.json'))['mlp']
out, imgs = {'T': T, 'd': float(Z['d']), 'R': int(Z['R'])}, {}
s_id, idx_id = term('score', 'cifar10_test'); c_id, _ = term('content', 'cifar10_test'); p_id, _ = term('pose', 'cifar10_test')
def pick(n, which=None):
    s, idx = term('score', n)
    i = int(np.argmin(np.abs(s - np.median(s)))) if which is None else which
    return i, s, idx
assert np.array_equal(Z['idx__fresh-cifar10_test'], Z['idx__fresh-cifar10_test__vflip'])
for key, n, data, pre in (('id', 'cifar10_test', 'cifar10_test', 'none'), ('vflip', 'cifar10_test__vflip', 'cifar10_test', 'vflip'), ('ood', 'tin_val', 'tin_val', 'none')):
    i, s, idx = pick(n, out['id']['pos'] if key == 'vflip' else None)
    c, _ = term('content', n); p, _ = term('pose', n)
    F = feats(f'fresh-{n}'); q = [F[l][i] for l in range(3)]
    sim = sum(B[l] @ q[l] for l in range(3)) / 3; nn = np.argsort(-sim)[:3]
    lg = np.load(f'{R}/runs/kx/rothead/s0/{tag}_k0_gqR/fresh-{n}.npz')['lg0_mlp'][i].astype(np.float64) / T
    post = np.exp(lg - lg.max()); post /= post.sum()
    img = np.load(f'{NPY}/{data}.npy', mmap_mode='r')[int(idx[i])]
    if pre == 'vflip': img = img[::-1]
    imgs[key] = np.asarray(img); imgs[key + '_nn'] = np.stack([np.asarray(bank_img[bidx[j]]) for j in nn])
    out[key] = dict(set=n, pos=i, data_index=int(idx[i]), score=float(s[i]), content=float(c[i]), pose=float(p[i]), post=post.tolist(),
                    nn_sim=[float(sim[j]) for j in nn], score_pct_in_id=float(100 * np.mean(s_id < s[i])),
                    content_pct_in_id=float(100 * np.mean(c_id < c[i])), pose_pct_in_id=float(100 * np.mean(p_id < p[i])))
np.savez(f'{R}/runs/kx/fig1_examples.npz', **imgs); json.dump(out, open(f'{R}/runs/kx/fig1_examples.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
