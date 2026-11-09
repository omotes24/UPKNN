# mk_fig2_en.py -- Figure 2 of the IJCAI draft (English version of the resume's mk_figs.py): the joint distribution of the two terms of s(x)
#   on the final evaluation images (noise seed 0) for the two ID sets with the lowest AUROC (CIFAR-10, ImageNet-64):
#   x = kNN term -d log r_R(x), y = orientation term log q(R_0 | x); ID (navy), upside-down ID (red), natural OOD sets pooled (grey).
#   Dotted / dash-dotted line: threshold of the kNN / orientation term alone that keeps 95% of the ID images; dashed line: the same for
#   the sum s(x) (x + y = const).
#   Default axes and ticks.   input: gen/data/terms_final2_s0.npz;  writes ../figs/fig_terms.pdf
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts():
    if 'NimbusRoman-Regular' in f or 'NimbusRoman-Italic' in f: font_manager.fontManager.addfont(f)
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.dirname(HERE); D = os.path.join(HERE, 'data')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Nimbus Roman', 'DejaVu Serif'], 'pdf.fonttype': 42, 'font.size': 8,
                     'axes.linewidth': 0.6, 'mathtext.fontset': 'stix'})
Z = np.load(f'{D}/terms_final2_s0.npz')
NAT = {'edmc10': ['svhn_test', 'textures', 'celeba_test', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val', 'cifar100_test'],
       'in64': ['in64_ssb_hard', 'in64_ninco', 'in64_inaturalist', 'in64_textures', 'in64_openimage_o', 'in64_places365', 'in64_celeba_test']}
IDN = {'edmc10': 'cifar10_test', 'in64': 'in64_imagenet_test'}
TITLE = {'edmc10': 'ID: CIFAR-10', 'in64': 'ID: ImageNet-64'}
NAVY, RED, GREY = '#1f3a7a', '#c0392b', '#9a9a9a'
fig, axs = plt.subplots(1, 2, figsize=(5.0, 1.9))
for ax, t in zip(axs, NAT):
    g = lambda k, n: Z[f'{t}|{k}|{n}'].astype(np.float64)
    ci, pi = g('content', IDN[t]), g('pose', IDN[t]); cv, pv = g('content', IDN[t] + '__vflip'), g('pose', IDN[t] + '__vflip')
    co = np.concatenate([g('content', o) for o in NAT[t]]); po = np.concatenate([g('pose', o) for o in NAT[t]])
    ax.scatter(co, po, s=1.5, color=GREY, alpha=0.35, linewidths=0, label='natural OOD', rasterized=True)
    ax.scatter(cv, pv, s=2.5, color=RED, alpha=0.7, linewidths=0, label='ID, upside down', rasterized=True)
    ax.scatter(ci, pi, s=2.5, color=NAVY, alpha=0.8, linewidths=0, label='ID', rasterized=True)
    tk = np.percentile(ci, 5); tp = np.percentile(pi, 5); ts = np.percentile(ci + pi, 5)
    ax.axvline(tk, color='k', lw=0.7, ls=':', label='threshold, kNN term only')
    ax.axhline(tp, color='k', lw=0.7, ls='-.', label='threshold, orientation term only')
    xl = np.array(ax.get_xlim()); ax.plot(xl, ts - xl, color='k', lw=0.8, ls='--', label='threshold, Upright kNN (sum)', scalex=False, scaley=False)
    ax.set_title(TITLE[t], fontsize=8.5); ax.set_xlabel('kNN term $-d\\,\\log r_R(x)$')
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
axs[0].set_ylabel('orientation term\n$\\log q_\\phi(R_0\\mid x)$')
h, l = axs[0].get_legend_handles_labels(); order = [2, 3, 1, 4, 0, 5]   # column-major fill: row 1 = ID, upside down, natural OOD; row 2 = the three thresholds
fig.legend([h[i] for i in order], [l[i] for i in order], loc='lower center', ncol=3, frameon=False, fontsize=6.8, markerscale=3, handletextpad=0.3, columnspacing=0.8)
fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0.2, 1, 1))
os.makedirs(f'{OUT}/figs', exist_ok=True); fig.savefig(f'{OUT}/figs/fig_terms.pdf', dpi=300); fig.savefig(f'{OUT}/figs/fig_terms.png', dpi=150)
print('written')
