# mk_fig3_en.py -- Figure 3: natural OOD images of ImageNet-64 flagged by one notion only (final evaluation images, noise seed 0), chosen by
#   the fixed rule of analysis/final/ana_two/qual/select_qual.py: for each OOD set except CelebA, the image flagged by the kNN term alone
#   (percentile < 5% of ID) that the orientation term finds most ID-like (top row), and the reverse (bottom row).  Under each image:
#   percentile of the kNN term / of the orientation term among the ID images.   inputs: gen/data/qual_ims.npz, qual_sel.json
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NimbusRoman-Regular' in f or 'NimbusRoman-Italic' in f: font_manager.fontManager.addfont(f)
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.dirname(HERE); D = os.path.join(HERE, 'data')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Nimbus Roman', 'DejaVu Serif'], 'pdf.fonttype': 42, 'font.size': 7})
Z = np.load(f'{D}/qual_ims.npz'); S = json.load(open(f'{D}/qual_sel.json'))
SN = {'in64_ssb_hard': 'SSB-hard', 'in64_ninco': 'NINCO', 'in64_inaturalist': 'iNaturalist', 'in64_textures': 'Textures',
      'in64_openimage_o': 'OpenImage-O', 'in64_places365': 'Places365'}
rows = [('knn_only', 'kNN term only'), ('orient_only', 'orientation term only')]
ims = {(r[0], int(r[2])): Z['ims'][k] for k, r in enumerate(zip(Z['names'], Z['names'], Z['idx']))}
fig, axs = plt.subplots(2, 6, figsize=(3.35, 1.62))
for i, (key, lab) in enumerate(rows):
    for j, (o, arr, idx, pc, pp) in enumerate(S[key]):
        ax = axs[i, j]; ax.imshow(ims[(o, idx)], interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_linewidth(0.4)
        fp = lambda v: f'{100 * v:.1f}' if 100 * v < 10 else f'{100 * v:.0f}'
        ax.set_xlabel(f'{fp(pc)} / {fp(pp)}', fontsize=6.2, labelpad=1.5)
        if i == 0: ax.set_title(SN[o], fontsize=6.2, pad=2)
    axs[i, 0].set_ylabel(lab.replace(' term only', '\nterm only'), fontsize=6.2, labelpad=2)
fig.tight_layout(pad=0.15, w_pad=0.25, h_pad=0.3)
os.makedirs(f'{OUT}/figs', exist_ok=True); fig.savefig(f'{OUT}/figs/fig_qual.pdf', dpi=300); fig.savefig(f'{OUT}/figs/fig_qual.png', dpi=200)
print('written')
