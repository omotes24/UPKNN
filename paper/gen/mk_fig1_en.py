# mk_fig1_en.py -- Figure 1(b) of the IJCAI draft (English labels; same data and layout rule as the resume's mk_fig1.py): three examples
#   (CIFAR-10 as ID, development split, noise seed 0, chosen by a fixed rule: the ID image with the median score, the same image upside down,
#   the Tiny-ImageNet image with the median score of its set) with the three nearest bank images, the percentile of the kNN term and of the
#   score among the ID images, the orientation posterior of the MLP, and the decision at the threshold keeping 95% of the ID images.
#   input: gen/data/fig1_examples.{json,npz};  writes ../figs/ex_*.png and ../figs/fig_examples.tex
import os, json
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.dirname(HERE); D = os.path.join(HERE, 'data')
J = json.load(open(f'{D}/fig1_examples.json')); Z = np.load(f'{D}/fig1_examples.npz')
os.makedirs(f'{OUT}/figs', exist_ok=True)
def save(a, name): Image.fromarray(np.asarray(a, np.uint8)).resize((128, 128), Image.NEAREST).save(f'{OUT}/figs/{name}.png')
for k in ('id', 'vflip', 'ood'):
    save(Z[k], f'ex_{k}')
    for j in range(3): save(Z[f'{k}_nn'][j], f'ex_{k}_nn{j}')
def pct(v): return f'{v:.0f}' if v >= 10 else f'{v:.1f}'
ROWS = [('id', 'ID', ''), ('vflip', 'ID, upside', 'down'), ('ood', 'OOD (Tiny-', 'ImageNet)')]
W, Y0, DY = 1.7, 0.0, 1.32
X = dict(lab=0.0, img=1.75, nn=2.95, knn=5.75, post=8.55, sc=11.15, dec=13.45)
L = ['% written by gen/mk_fig1_en.py (Figure 1(b))', '\\begin{tikzpicture}[font=\\scriptsize]',
     f'\\node[anchor=south] at ({X["img"]},{Y0 + 0.62}) {{input $\\mathbf{{x}}$}};',
     f'\\node[anchor=south] at ({X["nn"] + 0.9},{Y0 + 0.62}) {{nearest bank images}};',
     f'\\node[anchor=south, align=center] at ({X["knn"] + W / 2},{Y0 + 0.62}) {{kNN term\\\\[-1pt]{{\\tiny(percentile among ID)}}}};',
     f'\\node[anchor=south, align=center] at ({X["post"] + 0.79},{Y0 + 0.62}) {{posterior $q_\\phi(\\Rot{{r}}\\mid\\mathbf{{x}})$\\\\[-1pt]{{\\tiny($r$ in degrees)}}}};',
     f'\\node[anchor=south, align=center] at ({X["sc"] + W / 2},{Y0 + 0.62}) {{score $s(\\mathbf{{x}})$\\\\[-1pt]{{\\tiny(percentile among ID)}}}};',
     f'\\node[anchor=south] at ({X["dec"]},{Y0 + 0.62}) {{decision}};']
for i, (k, a, b) in enumerate(ROWS):
    e = J[k]; y = Y0 - i * DY
    L.append(f'\\node[align=center] at ({X["lab"]},{y}) {{{a}\\\\{b}}};' if b else f'\\node at ({X["lab"]},{y}) {{{a}}};')
    L.append(f'\\node[inner sep=0] at ({X["img"]},{y}) {{\\includegraphics[width=1.0cm]{{figs/ex_{k}.png}}}};')
    for j in range(3):
        L.append(f'\\node[inner sep=0] at ({X["nn"] + 0.1 + 0.85 * j},{y}) {{\\includegraphics[width=0.78cm]{{figs/ex_{k}_nn{j}.png}}}};')
    for key, x0, col in (('content_pct_in_id', X['knn'], 'navyc'), ('score_pct_in_id', X['sc'], 'black')):
        v = e[key]
        L.append(f'\\draw[line width=0.4pt] ({x0},{y - 0.11}) rectangle ({x0 + W},{y + 0.11});')
        L.append(f'\\fill[{col}] ({x0},{y - 0.11}) rectangle ({x0 + W * v / 100:.3f},{y + 0.11});')
        L.append(f'\\node[anchor=north] at ({x0 + W / 2},{y - 0.12}) {{\\tiny {pct(v)}\\%}};')
    for r, pr in enumerate(e['post']):
        xr = X['post'] + 0.46 * r
        L.append(f'\\fill[{"redc" if r == 0 else "black!45"}] ({xr},{y - 0.42}) rectangle ({xr + 0.2},{y - 0.42 + 0.84 * pr:.3f});')
        L.append(f'\\node[anchor=north, inner sep=1pt] at ({xr + 0.1},{y - 0.42}) {{\\tiny {[0, 90, 180, 270][r]}}};')
    L.append(f'\\draw[line width=0.3pt] ({X["post"] - 0.08},{y - 0.42}) -- ({X["post"] + 1.66},{y - 0.42});')
    ood = e['score_pct_in_id'] < 5
    L.append(f'\\node[text={"redc" if ood else "navyc"}] at ({X["dec"]},{y}) {{\\textbf{{{"OOD" if ood else "ID"}}}}};')
L.append('\\end{tikzpicture}')
open(f'{OUT}/figs/fig_examples.tex', 'w').write('\n'.join(L) + '\n')
print({k: (J[k]['content_pct_in_id'], J[k]['score_pct_in_id'], [round(p, 3) for p in J[k]['post']]) for k in ('id', 'vflip', 'ood')})
