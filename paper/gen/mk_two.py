# mk_two.py -- numbers_two.tex, tab_two.tex (Table 2) and supp_two.tex for the two-notions analysis (post-hoc; no change to the method)
#   inputs (gen/data): two_notions.json (analysis/final/ana_two/two_notions.py: per-image statistics of the two terms, final2, seeds 0-2),
#   two_subspace2/<tag>_s<seed>_<split>.json (code/two_subspace2.py on the GPU server: geometry and projected kNN; final2 seeds 0-2,
#   fresh seeds 0-2; the fitting variants for seed 0 on final2).
import os, json, glob
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.dirname(HERE); D = os.path.join(HERE, 'data')
TAGS = ['edmc10', 'l6svhn', 'dpceleba', 'edmc100', 'in64', 'fmnist']
NAME = {'edmc10': 'CIFAR-10', 'l6svhn': 'SVHN', 'dpceleba': 'CelebA', 'edmc100': 'CIFAR-100', 'in64': 'ImageNet-64', 'fmnist': 'FMNIST'}
TN = json.load(open(f'{D}/two_notions.json'))
TS = {}
for f in glob.glob(f'{D}/two_subspace2/*.json'):
    z = json.load(open(f)); TS[(z['tag'], z['seed'], z['split'])] = z
DEF = 'n4000_r1e-3'                                                          # the fitting of two_subspace.py
SEEDS = (0, 1, 2)


def strip(s, sp): return s.replace(f'{sp}-', '')
def nat(t, sp='final2'): return [strip(s, sp) for s in TS[(t, 0, sp)]['ood']]
def vf(t, sp='final2'): return strip(TS[(t, 0, sp)]['vflip'], sp)
def au(v): return v['mean'] if isinstance(v, dict) else v


def geo(t, sp='final2', var=DEF, seeds=SEEDS):
    """per-seed values, then mean over seeds"""
    r = {k: [] for k in ('ratio', 'btot', 'lda', 'pca', 'f3vf', 'p3vf', 'r3vf', 'f3nat', 'p3nat', 'r3nat', 'f1vf', 'f1nat', 'f3cvf', 'f3cnat', 'p3cnat', 'r3cnat')}
    for sd in seeds:
        z = TS.get((t, sd, sp))
        if z is None or var not in z['subspace']: continue
        S = z['subspace'][var]; K = z['knn_projection'][var]; rnd = S['random']['3']['mean']; N = nat(t, sp); V = vf(t, sp)
        r['ratio'].append(z['ratio_rot180']['median']); r['btot'].append(S['between_over_total'])
        r['lda'].append(S['lda']['3'] / rnd); r['pca'].append(S['pca']['3'] / rnd)
        r['f3vf'].append(K['lda3_euc'][V]); r['f3nat'].append(np.mean([K['lda3_euc'][o] for o in N]))
        r['f3cvf'].append(K['lda3_cos'][V]); r['f3cnat'].append(np.mean([K['lda3_cos'][o] for o in N]))
        if 'pca3_euc' in K:
            r['p3vf'].append(K['pca3_euc'][V]); r['p3nat'].append(np.mean([K['pca3_euc'][o] for o in N]))
            r['r3vf'].append(au(K['rand3_euc'][V])); r['r3nat'].append(np.mean([au(K['rand3_euc'][o]) for o in N]))
            r['f1vf'].append(K['lda1_euc'][V]); r['f1nat'].append(np.mean([K['lda1_euc'][o] for o in N]))
            r['p3cnat'].append(np.mean([K['pca3_cos'][o] for o in N])); r['r3cnat'].append(np.mean([au(K['rand3_cos'][o]) for o in N]))
    return {k: (float(np.mean(v)) if v else float('nan')) for k, v in r.items()} | {'n': len(r['ratio']), 'sd_f3nat': float(np.std(r['f3nat'])) if r['f3nat'] else float('nan')}


def tn(t):
    N = nat(t); V = vf(t); d = TN[t]
    m = lambda k: float(np.mean([d[o][k] for o in N]))
    return dict(knn_vf=d[V]['au_c'], head_vf=d[V]['au_p'], knn_nat=m('au_c'), head_nat=m('au_p'), sum_nat=m('au_s'), rho=m('rho'), rho_id=d['__id__']['rho'],
                only_c=m('only_c'), only_p=m('only_p'), both=m('both'), catch=float(np.nanmean([d[o]['miss_c_caught_p'] for o in N])))


G = {t: geo(t) for t in TAGS}; T = {t: tn(t) for t in TAGS}
f1 = lambda x: f'{x:.1f}'; f2 = lambda x: f'{x:.2f}'
def fx(x): return f'{x:.2f}' if x < 1 else (f'{x:.1f}' if x < 10 else f'{x:.0f}')
# ---------------------------------------------------------------------------------------------------------------- Table 2 rows
rows = []
for t in TAGS:
    g, n = G[t], T[t]
    rows.append(' & '.join([NAME[t], f2(g['ratio']), f1(100 * g['btot']), fx(g['lda']), fx(g['pca']), f1(n['knn_vf']), f1(g['p3vf']), f1(g['f3vf']), f1(n['head_vf']),
                            f1(n['knn_nat']), f1(n['head_nat']), f1(n['sum_nat']), f2(n['rho'])]) + r'\\')
open(f'{OUT}/tab_two.tex', 'w').write('% written by gen/mk_two.py\n' + '\n'.join(rows) + '\n')
# ---------------------------------------------------------------------------------------------------------------- macros
M = {}
M['ratioIN'] = f2(G['in64']['ratio']); M['ratioCH'] = f2(G['edmc100']['ratio']); M['ratioFM'] = f2(G['fmnist']['ratio'])
M['ratioCten'] = f2(G['edmc10']['ratio'])
M['vfINknn'] = f1(T['in64']['knn_vf']); M['vfCHknn'] = f1(T['edmc100']['knn_vf'])
rr = spearmanr([G[t]['ratio'] for t in TAGS], [T[t]['knn_vf'] for t in TAGS])[0]; M['rhoRatioFlip'] = f2(rr)
M['rhoBtotFlip'] = f2(spearmanr([G[t]['btot'] for t in TAGS], [T[t]['knn_vf'] for t in TAGS])[0])
five = [t for t in TAGS if t != 'dpceleba']
M['varLdaMin'] = fx(min(G[t]['lda'] for t in five)); M['varLdaMax'] = fx(max(G[t]['lda'] for t in five)); M['varLdaCelebA'] = fx(G['dpceleba']['lda'])
M['varLdaAllMin'] = fx(min(G[t]['lda'] for t in TAGS)); M['varLdaAllMax'] = fx(max(G[t]['lda'] for t in TAGS))
M['varPcMin'] = fx(min(G[t]['pca'] for t in TAGS)); M['varPcMax'] = fx(max(G[t]['pca'] for t in TAGS))
M['btotMin'] = f1(100 * min(G[t]['btot'] for t in TAGS)); M['btotMax'] = f1(100 * max(G[t]['btot'] for t in TAGS))
for k, nm in (('f3vf', 'vfLda'), ('p3vf', 'vfPc'), ('r3vf', 'vfRand'), ('f3nat', 'natLda'), ('p3nat', 'natPc'), ('r3nat', 'natRand')):
    M[nm + 'Min'] = f1(min(G[t][k] for t in TAGS)); M[nm + 'Max'] = f1(max(G[t][k] for t in TAGS)); M[nm + 'Mean'] = f1(np.mean([G[t][k] for t in TAGS]))
M['natLdaCten'] = f1(G['edmc10']['f3nat']); M['natPcCten'] = f1(G['edmc10']['p3nat']); M['natLdaIN'] = f1(G['in64']['f3nat']); M['natPcIN'] = f1(G['in64']['p3nat'])
M['nLdaBeatsPcNat'] = str(sum(G[t]['f3nat'] > G[t]['p3nat'] for t in TAGS)); M['nLdaBeatsPcVf'] = str(sum(G[t]['f3vf'] > G[t]['p3vf'] for t in TAGS))
M['tnRhoOODmin'] = f2(min(T[t]['rho'] for t in TAGS)); M['tnRhoOODmax'] = f2(max(T[t]['rho'] for t in TAGS))
M['tnRhoIDmin'] = f2(min(T[t]['rho_id'] for t in TAGS)); M['tnRhoIDmax'] = f2(max(T[t]['rho_id'] for t in TAGS))
i = T['in64']; kc, kp = i['only_c'] + i['both'], i['only_p'] + i['both']
M['tnINknn'] = f1(kc); M['tnINhead'] = f1(kp); M['tnINboth'] = f1(i['both']); M['tnINindep'] = f1(kc * kp / 100)
c = T['edmc10']; kc, kp = c['only_c'] + c['both'], c['only_p'] + c['both']
M['tnCtenknn'] = f1(kc); M['tnCtenhead'] = f1(kp); M['tnCtenboth'] = f1(c['both']); M['tnCtenindep'] = f1(kc * kp / 100)
hard = ['edmc10', 'l6svhn', 'edmc100', 'in64']
M['tnCatchMin'] = f1(min(T[t]['catch'] for t in hard)); M['tnCatchMax'] = f1(max(T[t]['catch'] for t in hard))
M['poseCtenCeleb'] = f1(TN['edmc10']['celeba_test']['au_p']); M['contentCtenCeleb'] = f1(TN['edmc10']['celeba_test']['au_c'])
M['contentChCten'] = f1(TN['edmc100']['cifar10_test']['au_c'])
# robustness (appendix): fitting variants (seed 0, final2), development split (seeds 0-2), seed spread
VARS = ['n4000_r1e-3', 'n4000_r1e-1', 'n10000_r1e-3', 'n10000_r1e-1']
VG = {t: {v: geo(t, var=v, seeds=(0,)) for v in VARS} for t in TAGS}
DV = {t: geo(t, sp='fresh') for t in TAGS}
M['varLdaVarMin'] = fx(min(VG[t][v]['lda'] for t in TAGS for v in VARS)); M['varLdaVarMax'] = fx(max(VG[t][v]['lda'] for t in TAGS for v in VARS))
M['natLdaVarMin'] = f1(min(VG[t][v]['f3nat'] for t in TAGS for v in VARS)); M['natLdaVarMax'] = f1(max(VG[t][v]['f3nat'] for t in TAGS for v in VARS))
M['sdLdaNatMax'] = f1(max(G[t]['sd_f3nat'] for t in TAGS))
po = [VG[t][v]['pca'] / VG[t][v]['lda'] for t in TAGS for v in VARS]
def big(x): return f'{x:,.0f}'.replace(',', '{,}') if x >= 100 else f'{x:.0f}'
M['pcOverLdaMin'] = big(min(po)); M['pcOverLdaMax'] = big(max(po))
H3 = ['edmc10', 'edmc100', 'in64']
M['vfLdaHardMin'] = f1(min(G[t]['f3vf'] for t in H3)); M['vfLdaHardMax'] = f1(max(G[t]['f3vf'] for t in H3))
M['vfPcHardMin'] = f1(min(G[t]['p3vf'] for t in H3)); M['vfPcHardMax'] = f1(max(G[t]['p3vf'] for t in H3))
M['vfKnnHardMin'] = f1(min(T[t]['knn_vf'] for t in H3)); M['vfKnnHardMax'] = f1(max(T[t]['knn_vf'] for t in H3))
M['vfHeadMin'] = f1(min(T[t]['head_vf'] for t in TAGS)); M['vfHeadMax'] = f1(max(T[t]['head_vf'] for t in TAGS))
M['vfLdaSVHN'] = f1(G['l6svhn']['f3vf'])
M['nSumBeatsBoth'] = str(sum(T[t]['sum_nat'] >= max(T[t]['knn_nat'], T[t]['head_nat']) - 1e-9 for t in TAGS))
M['btotIN'] = f1(100 * G['in64']['btot']); M['btotCH'] = f1(100 * G['edmc100']['btot'])
M['ldaShareMaxPct'] = f1(100 * max(TS[(t, 0, 'final2')]['subspace'][v]['lda']['3'] for t in TAGS for v in VARS))
M['pcShareMinPct'] = f'{100 * min(TS[(t, 0, "final2")]["subspace"][DEF]["pca"]["3"] for t in TAGS):.0f}'
M['pcShareMaxPct'] = f'{100 * max(TS[(t, 0, "final2")]["subspace"][DEF]["pca"]["3"] for t in TAGS):.0f}'
with open(f'{OUT}/numbers_two.tex', 'w') as f:
    f.write('% written by gen/mk_two.py\n')
    for k, v in M.items(): f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
# ---------------------------------------------------------------------------------------------------------------- appendix table
rows = []
for t in TAGS:
    cells = [NAME[t]]
    for v in VARS: g = VG[t][v]; cells += [fx(g['lda']), f1(g['f3vf']), f1(g['f3nat'])]
    d = DV[t]; cells += [f2(d['ratio']), f1(d['f3vf']), f1(d['f3nat']), f1(d['p3nat'])]
    rows.append(' & '.join(cells) + r'\\')
open(f'{OUT}/supp_two.tex', 'w').write('% written by gen/mk_two.py\n' + '\n'.join(rows) + '\n')
rows = [' & '.join([NAME[t], f1(G[t]['f3nat']), f1(G[t]['f3cnat']), f1(G[t]['p3nat']), f1(G[t]['p3cnat']), f1(G[t]['r3nat']), f1(G[t]['r3cnat']),
                    f1(T[t]['knn_nat']), f1(T[t]['head_nat'])]) + r'\\' for t in TAGS]
open(f'{OUT}/supp_two_nat.tex', 'w').write('% written by gen/mk_two.py\n' + '\n'.join(rows) + '\n')
SETN = {'cifar10_test': 'CIFAR-10', 'cifar100_test': 'CIFAR-100', 'svhn_test': 'SVHN', 'celeba_test': 'CelebA', 'textures': 'Textures',
        'tin_val': 'Tiny-ImageNet', 'lsun_fix': 'LSUN', 'imagenet_fix': 'ImageNet', 'imagenet32_val': 'ImageNet32', 'places365_val': 'Places365',
        'in64_ssb_hard': 'SSB-hard', 'in64_ninco': 'NINCO', 'in64_inaturalist': 'iNaturalist', 'in64_textures': 'Textures', 'in64_openimage_o': 'OpenImage-O',
        'in64_places365': 'Places365', 'in64_celeba_test': 'CelebA', 'g32_mnist_test': 'MNIST', 'g32_kmnist_test': 'KMNIST', 'g32_omniglot_eval': 'Omniglot',
        'g32_notmnist': 'notMNIST'}
rows = []
for t in TAGS:
    for k, o in enumerate(nat(t) + [vf(t)]):
        v = TN[t][o]; lab = SETN.get(o, 'upside-down ID')
        rows.append(' & '.join([NAME[t] if k == 0 else '', lab, f1(v['au_c']), f1(v['au_p']), f1(v['au_s']), f1(v['only_c']), f1(v['only_p']), f1(v['both']), f2(v['rho'])]) + r'\\')
    rows.append(r'\midrule')
open(f'{OUT}/supp_two_sets.tex', 'w').write('% written by gen/mk_two.py\n' + '\n'.join(rows[:-1]) + '\n')
print(json.dumps({t: {k: round(v, 3) for k, v in G[t].items()} for t in TAGS}, indent=0)[:3000])
print(M)
