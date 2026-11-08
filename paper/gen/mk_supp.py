# mk_supp.py -- tables of the supplementary material (English) from the same evaluation files as mk_ijcai.py (gen/data/):
#   supp_models.tex (models and configuration per ID), supp_pairs.tex (all 63 pairs), supp_fpr.tex (FPR95 per ID and method),
#   supp_ablate.tex (development-split ablations + selection data), supp_mirror.tex (mirror test by depth), supp_in64.tex (ImageNet-1K 64x64
#   per set), supp_in256.tex (uncontrolled 256x256 tier incl. SSB-hard / ImageNet-C), supp_eurosat.tex (EuroSAT per set).
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.dirname(HERE); D = os.path.join(HERE, 'data')
TAGS = ['edmc10', 'l6svhn', 'dpceleba', 'edmc100', 'in64', 'fmnist']
NAME = {'edmc10': 'CIFAR-10', 'l6svhn': 'SVHN', 'dpceleba': 'CelebA', 'edmc100': 'CIFAR-100', 'in64': 'ImageNet-64', 'fmnist': 'FashionMNIST'}
SETN = {'svhn_test': 'SVHN', 'textures': 'Textures', 'celeba_test': 'CelebA', 'tin_val': 'Tiny-ImageNet', 'lsun_fix': 'LSUN-FIX', 'imagenet_fix': 'ImageNet-FIX',
        'imagenet32_val': 'ImageNet32', 'places365_val': 'Places365', 'cifar100_test': 'CIFAR-100', 'cifar10_test': 'CIFAR-10', 'in64_ssb_hard': 'SSB-hard',
        'in64_ninco': 'NINCO', 'in64_inaturalist': 'iNaturalist', 'in64_textures': 'Textures', 'in64_openimage_o': 'OpenImage-O', 'in64_places365': 'Places365',
        'in64_celeba_test': 'CelebA', 'g32_mnist_test': 'MNIST', 'g32_kmnist_test': 'KMNIST', 'g32_omniglot_eval': 'Omniglot', 'g32_notmnist': 'notMNIST',
        'synth_noise': 'uniform noise', 'synth_const': 'constant', 'g32_synth_noise': 'uniform noise', 'g32_synth_const': 'constant'}
NEWJ = json.load(open(f'{D}/newrows_final2.json')); NEWROWS = [r for r in ('Graham', 'LMD', 'Kamkari') if r in NEWJ['rows']]
CSI = json.load(open(f'{D}/csi_ana.json')) if os.path.exists(f'{D}/csi_ana.json') else None
PRIOR = ['NLL', 'IC-png', 'IC-j2k', 'TT', 'MSMA*', 'DiffPath', 'GEPC', 'EigenScore'] + NEWROWS + ['CFS', 'CFS-kNN', 'RotCls4'] + (['CSI'] if CSI else [])
SHORT = {'NLL': 'NLL', 'IC-png': 'IC-PNG', 'IC-j2k': 'IC-J2K', 'TT': 'TT', 'MSMA*': 'MSMA$^{*}$', 'DiffPath': 'DiffPath', 'GEPC': 'GEPC', 'EigenScore': 'EigenScore',
         'CFS': 'CFS', 'CFS-kNN': 'CFS-kNN', 'RotCls4': 'Rot.\\ pred.', 'kNN tuned': 'kNN (tuned)', 'kNN (CT)': 'kNN (basic)',
         'Graham': 'DDPM-OOD', 'LMD': 'LMD', 'Kamkari': 'LID+NLL', 'CSI': 'CSI'}
def f1(v): return f'{v:.1f}'
def sgn(v): return '0.0' if abs(v) < 0.05 else ('$+$' if v > 0 else '$-$') + f'{abs(v):.1f}'
J = json.load(open(f'{D}/ours_vs_all_final2.json'))
for t in J['pairs']:                                                    # merge the new rows (same images and bootstrap draws) and CSI
    for o, p in J['pairs'][t].items():
        for r in NEWROWS:
            if r in NEWJ['pairs'][t][o]['rows']: p['rows'][r] = NEWJ['pairs'][t][o]['rows'][r]
        if CSI and t in CSI['pairs'] and o in CSI['pairs'][t]:
            c = CSI['pairs'][t][o]; p['rows']['CSI'] = dict(auroc=c['csi'], diff=c['diff'], ci=c['ci'], verdict=c['verdict'], fpr95=c['csi_fpr95'])
    for r in NEWROWS:
        if r in NEWJ['ids'][t]['rows']: J['ids'][t]['rows'][r] = NEWJ['ids'][t]['rows'][r]
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
def nat_sets(t): return [o for o in J['pairs'][t] if not o.endswith('__vflip') and 'synth' not in o]
def vf_set(t): return [o for o in J['pairs'][t] if o.endswith('__vflip')][0]
def fpr_cells(r):
    if r in FP['ids'][TAGS[0]]: return [f1(FP['ids'][t][r]['fpr95']) for t in TAGS] + [f1(FP['mean'][r]['fpr95']), f1(FP['vflip_fpr95'][r])]
    ids = []
    for t in TAGS:
        v = [J['pairs'][t][o]['rows'].get(r, {}).get('fpr95') for o in nat_sets(t)]
        ids.append(float(np.mean(v)) if all(x is not None for x in v) else None)
    vf = [J['pairs'][t][vf_set(t)]['rows'].get(r, {}).get('fpr95') for t in TAGS]
    mean = float(np.mean([x for x in ids if x is not None])) if any(x is not None for x in ids) else None
    vfm = float(np.mean([x for x in vf if x is not None])) if any(x is not None for x in vf) else None
    return [('--' if x is None else f1(x)) for x in ids] + [('--' if mean is None else f1(mean)) + ('' if all(x is not None for x in ids) else '$^{\\S}$'), '--' if vfm is None else f1(vfm)]
L = []
for r in RW:
    lab = 'Upright kNN' if r == 'Ours' else SHORT[r]
    L.append(f"{lab} & " + ' & '.join(fpr_cells(r)) + '\\\\')
open(f'{OUT}/supp_fpr.tex', 'w').write('\n'.join(L) + '\n')
# ---- ablations (development split, seed 0)
AB = {t: json.load(open(f'{D}/ablate2_{t}.json')) for t in TAGS}
ROWS = [('ours', 'Upright kNN, Eq.~(2)'), ('content only', 'kNN term only'), ('orientation only', 'orientation term only'), ('no temperature', 'no temperature ($T=1$)'),
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
# ---- the deep-block ablation: final evaluation vs Upright kNN, the development-split breakdown and controls, the frozen choices
DEEP = json.load(open(f'{D}/vl_vs_uknn_final2.json')); DEEPC = json.load(open(f'{D}/vl_ctrl_fresh.json')); DEEPP = json.load(open(f'{D}/PROTOCOL_VL.json'))
SIGGRID = [0.01 * 3162.2777 ** (k / 15) for k in range(16)]            # the 16-point noise grid (sigma_1 = 0.01)
def sgn(v): return '0.0' if abs(v) < 0.05 else ('$+$' if v > 0 else '$-$') + f'{abs(v):.1f}'
DNFE = {t: 1 + (1 if (DEEPP[t]['kstar'] is not None and DEEPP[t]['wstar'] > 0) else 0) for t in TAGS}
L = ['% written by gen/mk_supp.py: the deep-block ablation on the final images (natural-OOD mean per ID dataset, three noise seeds)']
for t in TAGS:
    r = DEEP['ids'][t]; vs = [q['verdict'] for q in DEEP['pairs'][t].values()]
    L.append(f"{NAME[t]} & {f1(r['upright'])} & {f1(r['vl'])} & {sgn(r['diff'])} [{sgn(r['ci'][0])}, {sgn(r['ci'][1])}] & {vs.count('win')}/{vs.count('tie')}/{vs.count('loss')} & {DNFE[t]}\\\\")
L.append('\\midrule')
W = DEEP['wtl']; L.append(f"Mean & {f1(DEEP['mean_upright'])} & {f1(DEEP['mean_vl'])} & {sgn(DEEP['mean_vl'] - DEEP['mean_upright'])} & {W['wins']}/{W['ties']}/{W['losses']} & --\\\\")
open(f'{OUT}/supp_deep.tex', 'w').write('\n'.join(L) + '\n')
def dcm(r): return DEEPC['mean'][r]
def dcp(t, o, r): return DEEPC['pairs'][t][o][r]['auroc']
def dwt(d): return f"{d['wins']}/{d['ties']}/{d['losses']}" if d else '--'
DROWS = [('Upright', 'Upright kNN', '1'), ('VL', '+ deep-block term (frozen rule)', '1--2'), ('VL-sigma1', '\\quad single pass only ($a_0e_{\\sigma_1}$)', '1'),
         ('pose-only', '\\quad orientation weight raised instead', '1'), ('chain', 'Control 1: chain rule (conditional kNN)', '1'),
         ('VL+mean', 'Control 2: + third expert (channel means)', '1--2')]
L = ['% written by gen/mk_supp.py: development images, three noise seeds (analysis/final/ana_kx/vl_ctrl_ana.py)']
for r, lab, nfe in DROWS:
    cells = [f1(dcm(r)), f1(dcp('edmc10', 'cifar100_test', r)), f1(dcp('edmc100', 'cifar10_test', r)), f1(dcp('in64', 'in64_ssb_hard', r)), f1(dcp('in64', 'in64_ninco', r))]
    L.append(f"{lab} & " + ' & '.join(cells) + f" & {dwt(DEEPC['wtl'].get(r))} & {dwt(DEEPC['wtl_vs_vl'].get(r))} & {nfe}\\\\")
    if r == 'VL': L.append('\\midrule')
open(f'{OUT}/supp_deep_ctrl.tex', 'w').write('\n'.join(L) + '\n')
L = ['% written by gen/mk_supp.py: the frozen choices of the deep-block ablation (noise seed 0, as recorded in PROTOCOL_VL.json; mirror tests in half precision)']
for t in TAGS:
    q = DEEPP[t]; ks = q['kstar']; m0 = q['mirror_levels'][f"L{q['level']}"]; ms = q['mirror_sigma'].get(f'k{ks}')
    L.append(f"{NAME[t]} & 1/{2 ** q['level']} & {f1(m0)} & {q['R']} & {q['a0']} & $\\sigma_{{{ks + 1}}}={SIGGRID[ks]:.3f}$ & {f1(ms)} [{f1(q['mirror_star_ci'][0])}, {f1(q['mirror_star_ci'][1])}] & {q['wstar']:.2f} & {DNFE[t]}\\\\")
open(f'{OUT}/supp_deep_choices.tex', 'w').write('\n'.join(L) + '\n')
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

# ---- E4: chest X-ray and histology (PROTOCOL_E4.md), generators trained with rotations
E4 = json.load(open(f'{D}/e4_ana.json'))
TF = ['rot90', 'rot180', 'vflip', 'hflip']
GROUPS = {'chest': dict(id='g32_chest_test', modality=['g32_organa_test', 'g32_path_gray_test'], source=['g32_pneumonia_all'], natural=['g32_cifar10_gray', 'g32_svhn_gray', 'g32_mnist_test', 'g32_fmnist_test']),
          'path': dict(id='path_test', modality=['blood_all', 'derma_all', 'chest_rgb_test'], source=[], natural=['cifar10_test', 'cifar100_test', 'svhn_test', 'celeba_test', 'textures', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val'])}
E4LAB = {'Ours': 'Upright kNN', 'kNN term': '\\quad kNN term', 'Orientation term': '\\quad orient.\\ term', 'kNN basic': 'kNN (basic)', 'CFS': 'CFS (diag.)', 'CFS-kNN': 'CFS (kNN)', 'NLL': 'NLL', 'MSMA*': 'MSMA$^{*}$', 'DiffPath': 'DiffPath'}
for tag in ('chest', 'path'):
    G = GROUPS[tag]; R_ = E4[tag]['rows']; L = []
    for r in ('Ours', 'kNN term', 'Orientation term', 'kNN basic', 'CFS', 'CFS-kNN', 'NLL', 'MSMA*', 'DiffPath'):
        if r not in R_: continue
        v = R_[r]; cells = [f1(v['per_set'][f"{G['id']}__{t}"]) for t in TF] + [f1(v['per_set'][x]) for x in G['modality']] + [f1(v['per_set'][x]) for x in G['source']] + [f1(v['natural'])]
        w = '--' if r == 'Ours' else '{}/{}/{}'.format(*E4[tag]['boot'][r]['wtl'])
        L.append(f"{E4LAB[r]} & " + ' & '.join(cells) + f' & {w}\\\\')
    open(f'{OUT}/supp_e4_{tag}.tex', 'w').write('\n'.join(L) + '\n')
L = []
for tag, lab in (('c10n4', 'none'), ('c10x4', 'horizontal flips'), ('c10r25', 'rotations, $p=0.25$'), ('c10r50', 'rotations, $p=0.5$'), ('c10r100', 'rotations, $p=1$')):
    R_ = E4[tag]['rows']; acc = float(np.mean([h['acc'] for h in E4[tag]['heads'].values()]))
    L.append(f"{lab} & {f1(acc)} & " + ' & '.join(f"{f1(R_[r]['natural'])} & {f1(R_[r]['flip'])}" for r in ('Ours', 'kNN term', 'Orientation term', 'NLL', 'NLL + log P(up|x)')) + '\\\\')
open(f'{OUT}/supp_e4_rot.tex', 'w').write('\n'.join(L) + '\n')
# ---- weight sweep (development images and final images)
W = json.load(open(f'{D}/weight_sweep.json')); ws = W['fresh']['w']
L = ['$w$ & ' + ' & '.join(str(w) for w in ws) + '\\\\\\midrule']
for split, lab in (('fresh', 'development images'), ('final2', 'final images')):
    L.append(f"\\multicolumn{{{len(ws) + 1}}}{{@{{}}l}}{{\\emph{{{lab}}}}}\\\\")
    for key, lab2 in (('mean_nat', 'mean, natural OOD sets'), ('near_mean', 'mean, near-OOD pairs'), ('mean_vflip', 'mean, upside-down probe')):
        L.append(f"{lab2} & " + ' & '.join(f'{v:.2f}' for v in W[split][key]) + '\\\\')
    for t in TAGS: L.append(f"\\quad {NAME[t]} & " + ' & '.join(f1(v) for v in W[split]['ids'][t]['nat']) + '\\\\')
open(f'{OUT}/supp_sweep.tex', 'w').write('\n'.join(L) + '\n')
# ---- two-notions controls (PROTOCOL_E6.md), when the runs have finished: gen/data/two_controls/<tag>_s<seed>.json
import glob
TC = {}
for f in glob.glob(f'{D}/two_controls/*_s*.json'):
    z = json.load(open(f)); TC.setdefault(z['tag'], {})[z['seed']] = z
if TC:
    def tcm(tag, fn): return float(np.mean([fn(z) for z in TC[tag].values()]))
    L = []
    for t in TAGS:
        if t not in TC: continue
        c = [tcm(t, lambda z: z['A_full_head_space']['euc']['flip']), tcm(t, lambda z: z['A_full_head_space']['cos']['flip']), tcm(t, lambda z: z['A_full_head_space']['euc']['nat_mean']),
             tcm(t, lambda z: z['B_mean3']['share_over_random3']), tcm(t, lambda z: z['B_mean3']['knn']['euc']['flip']), tcm(t, lambda z: z['B_mean3']['knn']['euc']['nat_mean']),
             tcm(t, lambda z: 100 * z['C_head_g_only']['acc']), tcm(t, lambda z: z['C_head_g_only']['alone']['flip']), tcm(t, lambda z: z['C_head_g_only']['alone']['nat_mean']), tcm(t, lambda z: z['C_head_g_only']['sum_with_knn_term']['nat_mean']),
             tcm(t, lambda z: 100 * z['D_pixel_head']['pix2']['acc']), tcm(t, lambda z: z['D_pixel_head']['pix2']['alone']['flip']), tcm(t, lambda z: z['D_pixel_head']['pix2']['alone']['nat_mean']), tcm(t, lambda z: z['D_pixel_head']['pix2']['sum_with_knn_term']['nat_mean'])]
        L.append(f"{NAME[t]} & " + ' & '.join((f'{v:.0f}' if i == 3 else f1(v)) for i, v in enumerate(c)) + '\\\\')
    open(f'{OUT}/supp_controls.tex', 'w').write('\n'.join(L) + '\n')
    L = []
    for t in TAGS:
        if t not in TC: continue
        L.append(f"{NAME[t]} & " + ' & '.join(f1(tcm(t, lambda z, k=k: z['E_d'][k])) for k in ('n3000_k5', 'n3000_k10', 'n3000_k20', 'n3000_k50', 'n10000_k5', 'n10000_k10', 'n10000_k20', 'n10000_k50')) + '\\\\')
    open(f'{OUT}/supp_dsens.tex', 'w').write('\n'.join(L) + '\n')
# ---- ImageNet-64 bank size (PROTOCOL_E5B.md): gen/data/e5b_ana.json (analysis/final/e5b/ana/e5b_ana.py), when the runs have finished
E5B = json.load(open(f'{D}/e5b_ana.json')) if os.path.exists(f'{D}/e5b_ana.json') else None
if E5B:
    L = []
    for tag, lab in (('in64', '10{,}000 (10 per class; the paper)'), ('in64b25', '25{,}000 (random)'), ('in64b50', '50{,}000 (random)')):
        if tag not in E5B: continue
        L.append(f"\\midrule\\multicolumn{{10}}{{@{{}}l}}{{\\emph{{bank of {lab} training images}}}}\\\\")
        for r, rl in (('Ours (R=N/10)', '\\textbf{Upright kNN} ($R=N/10$)'), ('Ours (R=1000)', '\\quad $R=1000$'), ('kNN term', '\\quad kNN term ($R=N/10$)'), ('Orientation term', '\\quad orient.\\ term'),
                      ('kNN basic', 'kNN (basic)'), ('CFS', 'CFS (diag.)'), ('CFS kNN', 'CFS (kNN)')):
            if r not in E5B[tag]['rows']: continue
            v = E5B[tag]['rows'][r]
            L.append(rl + ' & ' + ' & '.join(f1(x) for x in (v['near'], v['far'], v['near_fpr95'], v['far_fpr95'], v['in64_imagenet_v2']['auroc'], v['in64_imagenet_c']['auroc'], v['in64_imagenet_r']['auroc'], v['FS-near'], v['FS-far'])) + '\\\\')
    open(f'{OUT}/supp_e5b.tex', 'w').write('% written by gen/mk_supp.py (PROTOCOL_E5B.md)\n' + '\n'.join(L) + '\n')
    MX = {}
    for tag, c in (('in64', 'Ten'), ('in64b25', 'TwoFive'), ('in64b50', 'Fifty')):
        if tag in E5B:
            o = E5B[tag]['rows']['Ours (R=N/10)']; k = E5B[tag]['rows']['kNN term']
            MX[f'bank{c}Near'] = f1(o['near']); MX[f'bank{c}Far'] = f1(o['far']); MX[f'bank{c}NearKnnT'] = f1(k['near']); MX[f'bank{c}FarKnnT'] = f1(k['far'])
            MX[f'bank{c}NearRk'] = f1(E5B[tag]['rows']['Ours (R=1000)']['near'])
    assert all(not any(ch.isdigit() for ch in k) for k in MX), [k for k in MX if any(ch.isdigit() for ch in k)]
    with open(f'{OUT}/numbers_e5b.tex', 'w') as fo:
        fo.write('% written by gen/mk_supp.py\n')
        for k, v in MX.items(): fo.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
# ---- the frame term as a plug-in (PROTOCOL_E9.md): gen/data/e9_ana.json (analysis/final/e9/ana/e9_ana.py), when the runs have finished
E9 = json.load(open(f'{D}/e9_ana.json')) if os.path.exists(f'{D}/e9_ana.json') else None
if E9:
    LAB9 = {'CFS (kNN)': 'CFS (kNN)', 'Upright CFS (kNN)': '\\quad + frame term (Bayes weight)', 'CFS (kNN) + frame (z)': '\\quad + frame term ($z$-sum)',
            'kNN (basic)': 'kNN (basic)', 'Upright kNN (basic)': '\\quad + frame term (Bayes weight)', 'kNN (basic) + frame (z)': '\\quad + frame term ($z$-sum)',
            'CFS (diag)': 'CFS (diag.)', 'CFS (diag, Gaussian)': 'CFS (diag.) as a Gaussian log-density', 'Upright CFS (diag)': '\\quad + frame term (unit weight)', 'CFS (diag) + frame (z)': '\\quad + frame term ($z$-sum)',
            'kNN term': 'kNN term of Upright kNN', 'Ours': '\\quad + frame term (Bayes weight; Upright kNN)',
            'NLL': 'NLL', 'IC-png': 'IC (PNG)', 'IC-j2k': 'IC (JPEG2K)', 'TT': 'TT', 'MSMA*': 'MSMA$^{*}$', 'DiffPath': 'DiffPath', 'GEPC': 'GEPC', 'EigenScore': 'EigenScore', 'RotCls4': 'Rotation pred.'}
    ORDER = [('kNN term', 'Ours', None), ('kNN (basic)', 'Upright kNN (basic)', 'kNN (basic) + frame (z)'), ('CFS (kNN)', 'Upright CFS (kNN)', 'CFS (kNN) + frame (z)'),
             ('CFS (diag)', None, 'CFS (diag) + frame (z)'), ('CFS (diag, Gaussian)', 'Upright CFS (diag)', None)] + [(m, None, f'{m} + frame (z)') for m in ('NLL', 'IC-png', 'IC-j2k', 'TT', 'MSMA*', 'DiffPath', 'GEPC', 'EigenScore', 'RotCls4')]
    def wtl9(up, base):
        k = f'{up} vs {base}'
        if k not in E9['wtl_nat']: return '--'
        w = E9['wtl_nat'][k]; b = E9['big_nat'][k]; return f"{w['win']}/{w['tie']}/{w['loss']} ({b['win1']}/{b['loss1']})"
    L = []
    for base, up, fz in ORDER:
        for r in (base, up, fz):
            if r is None or r not in E9['mean']: continue
            cells = [f1(E9['ids'][t][r]) if r in E9['ids'][t] else '--' for t in TAGS] + [f1(E9['mean'][r]), f1(E9['vflip_mean'][r]), '--' if r == base else wtl9(r, base)]
            L.append((LAB9[r] if r == base or r in ('Ours', 'Upright CFS (kNN)', 'Upright kNN (basic)', 'Upright CFS (diag)') else '\\quad + frame term ($z$-sum)') + ' & ' + ' & '.join(cells) + '\\\\')
        L.append('\\addlinespace[1pt]')
    open(f'{OUT}/supp_e9.tex', 'w').write('% written by gen/mk_supp.py (PROTOCOL_E9.md)\n' + '\n'.join(L) + '\n')
    MX = {}
    def gain(up, base): return E9['mean'][up] - E9['mean'][base]
    MX.update({'eNineCfsKnn': f1(E9['mean']['CFS (kNN)']), 'eNineUpCfsKnn': f1(E9['mean']['Upright CFS (kNN)']), 'eNineGainCfsKnn': f1(gain('Upright CFS (kNN)', 'CFS (kNN)')),
               'eNineCfsKnnWtl': wtl9('Upright CFS (kNN)', 'CFS (kNN)').split(' ')[0], 'eNineCfsKnnBigWin': str(E9['big_nat']['Upright CFS (kNN) vs CFS (kNN)']['win1']),
               'eNineCfsKnnLoss': str(E9['wtl_nat']['Upright CFS (kNN) vs CFS (kNN)']['loss']),
               'eNineKnnB': f1(E9['mean']['kNN (basic)']), 'eNineUpKnnB': f1(E9['mean']['Upright kNN (basic)']), 'eNineGainKnnB': f1(gain('Upright kNN (basic)', 'kNN (basic)')),
               'eNineKnnBWtl': wtl9('Upright kNN (basic)', 'kNN (basic)').split(' ')[0], 'eNineKnnBLoss': str(E9['wtl_nat']['Upright kNN (basic) vs kNN (basic)']['loss']),
               'eNineCfsDiag': f1(E9['mean']['CFS (diag)']), 'eNineCfsDiagG': f1(E9['mean']['CFS (diag, Gaussian)']), 'eNineUpCfsDiag': f1(E9['mean']['Upright CFS (diag)']), 'eNineGainCfsDiag': f"{gain('Upright CFS (diag)', 'CFS (diag, Gaussian)'):+.1f}",
               'eNineCfsDiagWtl': wtl9('Upright CFS (diag)', 'CFS (diag, Gaussian)').split(' ')[0],
               'eNineGainKnnT': f1(gain('Ours', 'kNN term')), 'eNineUpCfsKnnVf': f1(E9['vflip_mean']['Upright CFS (kNN)']), 'eNineCfsKnnVf': f1(E9['vflip_mean']['CFS (kNN)']),
               'eNineZCfsKnn': f1(E9['mean']['CFS (kNN) + frame (z)']), 'eNineZKnnB': f1(E9['mean']['kNN (basic) + frame (z)']),
               'eNineDcfsMin': f'{min(v["d_cfs"] for t in E9["space"] for v in E9["space"][t].values()):.0f}', 'eNineDcfsMax': f'{max(v["d_cfs"] for t in E9["space"] for v in E9["space"][t].values()):.0f}',
               'eNineDlMin': f'{min(v["d_l1"] for t in E9["space"] for v in E9["space"][t].values()):.0f}', 'eNineDlMax': f'{max(v["d_l1"] for t in E9["space"] for v in E9["space"][t].values()):.0f}'})
    zg = {m: gain(f'{m} + frame (z)', m) for m in ('NLL', 'IC-png', 'IC-j2k', 'TT', 'MSMA*', 'DiffPath', 'GEPC', 'EigenScore', 'RotCls4', 'CFS (diag)', 'CFS (kNN)', 'kNN (basic)') if f'{m} + frame (z)' in E9['mean']}
    MX['eNineZgainMin'] = f'{min(zg.values()):+.1f}'; MX['eNineZgainMax'] = f'{max(zg.values()):+.1f}'; MX['eNineZnImproved'] = str(sum(v > 0 for v in zg.values())); MX['eNineZn'] = str(len(zg))
    MX['eNineZworst'] = min(zg, key=zg.get); MX['eNineZbest'] = max(zg, key=zg.get)
    above = [m for m in zg if E9['mean'][f'{m} + frame (z)'] > E9['mean']['Orientation term']]
    MX['eNineZabove'] = ', '.join(LAB9[m] for m in above); MX['eNineZnAbove'] = str(len(above)); MX['eNineOrient'] = f1(E9['mean']['Orientation term'])
    MX['eNineZCfsDiag'] = f1(E9['mean']['CFS (diag) + frame (z)']); MX['eNineZNll'] = f1(E9['mean']['NLL + frame (z)']); MX['eNineNll'] = f1(E9['mean']['NLL'])
    MX['eNineUpKnnBVf'] = f1(E9['vflip_mean']['Upright kNN (basic)']); MX['eNineKnnBVf'] = f1(E9['vflip_mean']['kNN (basic)'])
    assert all(not any(ch.isdigit() for ch in k) for k in MX)
    with open(f'{OUT}/numbers_e9.tex', 'w') as fo:
        fo.write('% written by gen/mk_supp.py\n')
        for k, v in MX.items(): fo.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
print('E4 / sweep / controls tables written', bool(TC), 'E5B', bool(E5B), 'E9', bool(E9))
