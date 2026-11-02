# select_qual.py -- fixed rule (written before looking at any image): ImageNet-64, final evaluation images, noise seed 0.  For each natural
#   OOD set except CelebA (faces of public figures), the image flagged by the kNN term alone (kNN percentile < 5% of ID, orientation
#   percentile >= 5%) that the orientation term finds most ID-like, and the image flagged by the orientation term alone that the kNN term
#   finds most ID-like (ties: the more clearly flagged one).  Percentile = fraction of ID images with a lower value of that term.
import os, json, numpy as np
SCR = os.environ.get('UPKNN_ANALYSIS', os.path.expanduser('~/upknn/analysis'))
Z = np.load(f'{SCR}/final/data_kx/export/in64_s0_final2.npz')
SETS = {r[0]: (r[1], r[2]) for r in json.load(open(f'{SCR}/final/eu_stage/push_final2/final2_idx/sets_final2_in64.json'))}
OODS = ['in64_ssb_hard', 'in64_ninco', 'in64_inaturalist', 'in64_textures', 'in64_openimage_o', 'in64_places365']
g = lambda k, n: Z[f'{k}__final2-{n}'].astype(np.float64)
ci, pi = np.sort(g('content', 'in64_imagenet_test')), np.sort(g('pose', 'in64_imagenet_test'))
pct = lambda ref, v: np.searchsorted(ref, v, 'left') / len(ref)
out = {'knn_only': [], 'orient_only': [], 'counts': {}}
for o in OODS:
    c, p, idx = g('content', o), g('pose', o), Z[f'idx__final2-{o}']
    pc, pp = pct(ci, c), pct(pi, p)
    ko = np.where((pc < 0.05) & (pp >= 0.05))[0]; oo = np.where((pp < 0.05) & (pc >= 0.05))[0]
    out['counts'][o] = [int(len(ko)), int(len(oo)), int(len(idx))]
    if len(ko): j = ko[np.lexsort((pc[ko], -pp[ko]))[0]]; out['knn_only'].append([o, SETS[f'final2-{o}'][0], int(idx[j]), float(pc[j]), float(pp[j])])
    if len(oo): j = oo[np.lexsort((pp[oo], -pc[oo]))[0]]; out['orient_only'].append([o, SETS[f'final2-{o}'][0], int(idx[j]), float(pc[j]), float(pp[j])])
json.dump(out, open('qual_sel.json', 'w'), indent=1); print(json.dumps(out, indent=1))
