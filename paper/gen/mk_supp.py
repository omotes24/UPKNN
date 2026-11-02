# mk_supp.py -- tables of the supplementary material (English) from the same evaluation files as mk_ijcai.py (gen/data/):
#   supp_models.tex (models and configuration per ID), supp_pairs.tex (all 63 pairs), supp_fpr.tex (FPR95 per ID and method),
#   supp_ablate.tex (development-split ablations + selection data), supp_mirror.tex (mirror test by depth), supp_in64.tex (ImageNet-1K 64x64
#   per set), supp_in256.tex (uncontrolled 256x256 tier incl. SSB-hard / ImageNet-C), supp_eurosat.tex (EuroSAT per set).
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.dirname(HERE); D = os.path.join(HERE, 'data')
TAGS = ['edmc10', 'l6svhn', 'dpceleba', 'edmc100', 'in64', 'fmnist']
NAME = {'edmc10': 'CIFAR-10', 'l6svhn': 'SVHN', 'dpceleba': 'CelebA', 'edmc100': 'CIFAR-100', 'in64': 'ImageNet-64', 'fmnist': 'FashionMNIST'}
SETN = {'svhn_test': 'SVHN', 'textures': 'Textures', 'celeba_test': 'CelebA', 'tin_val': 'Tiny-ImageNet', 'lsun_fix': 'LSUN', 'imagenet_fix': 'ImageNet (resized)',
        'imagenet32_val': 'ImageNet32', 'places365_val': 'Places365', 'cifar100_test': 'CIFAR-100', 'cifar10_test': 'CIFAR-10', 'in64_ssb_hard': 'SSB-hard',
        'in64_ninco': 'NINCO', 'in64_inaturalist': 'iNaturalist', 'in64_textures': 'Textures', 'in64_openimage_o': 'OpenImage-O', 'in64_places365': 'Places365',
        'in64_celeba_test': 'CelebA', 'g32_mnist_test': 'MNIST', 'g32_kmnist_test': 'KMNIST', 'g32_omniglot_eval': 'Omniglot', 'g32_notmnist': 'notMNIST',
        'synth_noise': 'uniform noise', 'synth_const': 'constant', 'g32_synth_noise': 'uniform noise', 'g32_synth_const': 'constant'}
PRIOR = ['NLL', 'IC-png', 'IC-j2k', 'TT', 'MSMA*', 'DiffPath', 'GEPC', 'EigenScore', 'CFS', 'CFS-kNN', 'RotCls4']
SHORT = {'NLL': 'NLL', 'IC-png': 'IC-PNG', 'IC-j2k': 'IC-J2K', 'TT': 'TT', 'MSMA*': 'MSMA$^{*}$', 'DiffPath': 'DiffPath', 'GEPC': 'GEPC', 'EigenScore': 'EigenScore',
         'CFS': 'CFS', 'CFS-kNN': 'CFS-kNN', 'RotCls4': 'Rot.\\ pred.', 'kNN tuned': 'kNN (tuned)', 'kNN (CT)': 'kNN (basic)'}
def f1(v): return f'{v:.1f}'
def sgn(v): return '0.0' if abs(v) < 0.05 else ('$+$' if v > 0 else '$-$') + f'{abs(v):.1f}'
J = json.load(open(f'{D}/ours_vs_all_final2.json'))
# ---- models
MOD = {'edmc10': ('$32{\\times}32$', 'EDM (VP, DDPM++)', '\\cite{karras2022edm}'), 'l6svhn': ('$32{\\times}32$', 'VP-SDE', '\\cite{kamkari2024geometric}'),
       'dpceleba': ('$32{\\times}32$', 'iDDPM', '\\cite{nichol2021improved}'), 'edmc100': ('$32{\\times}32$', 'DDPM++ (EDM precond.)', '\\cite{chen2024robust}'),
       'in64': ('$64{\\times}64$', 'EDM2-XS (uncond.)', '\\cite{karras2024edm2}'), 'fmnist': ('$32{\\times}32$', 'U-Net (EDM precond.)', 'trained by us')}
ACC = {'edmc10': 95.5, 'l6svhn': 86.7, 'dpceleba': 100.0, 'edmc100': 88.2, 'in64': 77.3, 'fmnist': 99.7}
EXPD = json.load(open(f'{D}/dR_s0.json')); PS = json.load(open(f'{D}/ok_prof_sel.json')); PERID = {t: max(PS, key=lambda k: PS[k][t]) for t in TAGS}
def tuned_cfg(k):
    st, g, kk, r = k.split('_'); blk = 'mean of 3 blocks' if g == 'L012' else 'half-res.\\ block'
    return f"{blk}, $\\sigma_{{{1 if kk == 'k0' else 7}}}$, " + (f'{r[1:]}th neighbour' if st == 'P' else f'mean of top {r[1:]}')
L = []
for t in TAGS:
    a, b, c = MOD[t]
    L.append(f"{NAME[t]} & {a} & {b} & {c} & {float(EXPD[t]['d']):.1f} & {int(EXPD[t]['R'])} & {float(EXPD[t]['T']):.2f} & {ACC[t]:.1f} & {tuned_cfg(PERID[t])}\\\\")
open(f'{OUT}/supp_models.tex', 'w').write('\n'.join(L) + '\n')
# ---- all pairs
def dcell(x):
    d = sgn(x['diff']); return {'win': f'\\textbf{{{d}}}', 'loss': f'\\textit{{{d}}}', 'tie': d}[x['verdict']]
L = []
for t in TAGS:
    first = True
    for o, p in J['pairs'][t].items():
        R_ = p['rows']; bp = max([m for m in PRIOR if m in R_], key=lambda m: R_[m]['auroc'])
        on = SETN.get(o) or ('upside-down ID' if o.endswith('__vflip') else o)
        L.append(f"{NAME[t] if first else ''} & {on} & {f1(p['ours'])} & {f1(R_[bp]['auroc'])} & {SHORT[bp]} & {dcell(R_[bp])} & {f1(R_['kNN tuned']['auroc'])} & {dcell(R_['kNN tuned'])}\\\\")
        first = False
    L.append('\\midrule')
L[-1] = '\\bottomrule'; open(f'{OUT}/supp_pairs.tex', 'w').write('\n'.join(L) + '\n')
# ---- FPR95
FP = json.load(open(f'{D}/fpr95_all_final2.json')); RW = ['Ours', 'kNN tuned', 'kNN (CT)'] + PRIOR
L = []
for r in RW:
    lab = 'Upright kNN' if r == 'Ours' else SHORT[r]
    L.append(f"{lab} & " + ' & '.join(f1(FP['ids'][t][r]['fpr95']) for t in TAGS) + f" & {f1(FP['mean'][r]['fpr95'])} & {f1(FP['vflip_fpr95'][r])}\\\\")
open(f'{OUT}/supp_fpr.tex', 'w').write('\n'.join(L) + '\n')
# ---- ablations (development split, seed 0)
AB = {t: json.load(open(f'{D}/ablate2_{t}.json')) for t in TAGS}
ROWS = [('ours', 'Upright kNN, Eq.~(5)'), ('content only', 'kNN term only'), ('orientation only', 'orientation term only'), ('no temperature', 'no temperature ($T=1$)'),
        ('Fisher instead of sum', 'Fisher (sum of log $p$-values)'), ('orbit bank', 'rotated bank for $\\bar p$ ($4R$, $d$ re-estimated)'),
        ('head with rotated noise', 'head trained with rotated noise$^{*}$'), ('four views (rotated-noise test views)', 'orientation term over 4 test views$^{*}$')]
L = []
for k, lab in ROWS:
    nat = [AB[t][k]['nat'] for t in TAGS]
    L.append(f"{lab} & " + ' & '.join(f1(v) for v in nat) + f" & {f1(np.mean(nat))} & {f1(np.mean([AB[t][k]['vflip'] for t in TAGS]))} & {f1(np.mean([AB[t][k]['sel'] for t in TAGS]))}\\\\")
open(f'{OUT}/supp_ablate.tex', 'w').write('\n'.join(L) + '\n')
# ---- mirror test (rows of the resume's table, numbers only)
rows = [l for l in open(f'{D}/tab_mirror_rows.tex').read().split('\n') if l and not l.startswith('%')]
rows = [l.replace('\\bottomrule', '') for l in rows]
open(f'{OUT}/supp_mirror.tex', 'w').write('\n'.join(rows) + '\n')
# ---- ImageNet-1K 64x64 per set
O = json.load(open(f'{D}/ood1k_ana.json')); C = O['common']
SETS = ['in64_ssb_hard', 'in64_ninco', 'in64_inaturalist', 'in64_textures', 'in64_openimage_o', 'in64_imagenet_v2', 'in64_imagenet_c', 'in64_imagenet_r']
LAB = {'Ours': 'Upright kNN', 'kNN term': '\\quad kNN term', 'Orientation term': '\\quad orient.\\ term', 'VL': None, 'kNN basic': 'kNN (basic)', 'CFS': 'CFS (diag.)',
       'CFS kNN': 'CFS (kNN)', 'RotCls4': 'Rot.\\ pred.', 'NLL': 'NLL', 'IC-png': 'IC-PNG', 'IC-j2k': 'IC-J2K', 'TT': 'TT', 'MSMA*': 'MSMA$^{*}$', 'DiffPath': 'DiffPath',
       'GEPC': 'GEPC', 'EigenScore': 'EigenScore'}
L = []
for r, v in C.items():
    if LAB.get(r) is None: continue
    w = '--' if r == 'Ours' else '{}/{}/{}'.format(*O['boot'][r]['wtl'])
    L.append(f"{LAB[r]} & " + ' & '.join(f"{f1(v[s]['auroc'])}" for s in SETS) + f" & {f1(v['near_fpr95'])} & {f1(v['far_fpr95'])} & {w}\\\\")
open(f'{OUT}/supp_in64.tex', 'w').write('\n'.join(L) + '\n')
sev = O['imagenet_c']
L = [f"{LAB[r]} & " + ' & '.join(f1(sev[r]['severity'][k]) for k in '12345') + '\\\\' for r in ('Ours', 'kNN term', 'Orientation term', 'kNN basic', 'CFS')]
open(f'{OUT}/supp_in64c.tex', 'w').write('\n'.join(L) + '\n')
# ---- ImageNet 256 (uncontrolled)
H = json.load(open(f'{D}/in256_ana.json')); HC = H['common']
S256 = ['ssb_hard', 'ninco', 'inaturalist', 'textures', 'openimage_o', 'imagenet_test__vflip']
L = [f"{LAB[r]} & " + ' & '.join(f1(HC[r][s]['auroc']) for s in S256) + '\\\\' for r in ('Ours', 'kNN term', 'Orientation term', 'kNN basic', 'CFS', 'CFS kNN', 'MSMA*', 'DiffPath', 'GEPC')]
open(f'{OUT}/supp_in256.tex', 'w').write('\n'.join(L) + '\n')
# ---- EuroSAT
E = json.load(open(f'{D}/e3_ana.json')); ES = ['cifar10_test', 'cifar100_test', 'svhn_test', 'celeba_test', 'textures', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val']
L = [f"{ {'Ours': 'Upright kNN', 'kNN term': '\\quad kNN term', 'Orientation term': '\\quad orient.\\ term', 'kNN basic': 'kNN (basic)'}[r] } & " +
     ' & '.join(f1(E['rows'][r]['per_set'][s]) for s in ES) + f" & {f1(E['rows'][r]['mean'])} & {f1(E['rows'][r]['vflip'])} & {f1(E['rows'][r]['rot90'])}\\\\" for r in ('Ours', 'kNN term', 'Orientation term', 'kNN basic')]
open(f'{OUT}/supp_eurosat.tex', 'w').write('\n'.join(L) + '\n')
print('supplementary tables written')
