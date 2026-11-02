# mk_ijcai.py -- numbers and table bodies of the IJCAI-27 draft (English) from the evaluation files in gen/data/:
#   ours_vs_all_final2.json / _fresh.json (6 ID tiers, final split evaluated once after freezing / development split; ours_vs_all.py),
#   fpr95_all_final2.json, ablate2_<tag>.json (development split + selection data, seed 0), dR_s0.json, bench_cpu_session*.jsonl,
#   ood1k_ana.json (OpenOOD v1.5 ImageNet-1K at 64x64, ood1k_ana.py), fusion_final2.json (naive fusions, fusion_final2.py),
#   in256_ana.json (ImageNet 256x256 with ADM, in256_ana.py; optional until the runs finish).
#   writes ../numbers.tex, ../tab_main.tex, ../tab_imagenet.tex, ../tab_fusion.tex
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.dirname(HERE); D = os.path.join(HERE, 'data')
TAGS = ['edmc10', 'l6svhn', 'dpceleba', 'edmc100', 'in64', 'fmnist']
VF = {'edmc10': 'cifar10_test__vflip', 'l6svhn': 'svhn_test__vflip', 'dpceleba': 'celeba_test__vflip', 'edmc100': 'cifar100_test__vflip',
      'in64': 'in64_imagenet_test__vflip', 'fmnist': 'g32_fmnist_test__vflip'}
EIGHT = ['NLL', 'IC-png', 'IC-j2k', 'TT', 'MSMA*', 'DiffPath', 'GEPC', 'EigenScore']
PRIOR = EIGHT + ['CFS', 'CFS-kNN', 'RotCls4']
LABEL = {'NLL': 'NLL', 'IC-png': 'IC (PNG)', 'IC-j2k': 'IC (JPEG2000)', 'TT': 'TT', 'MSMA*': 'MSMA$^{*}$', 'DiffPath': 'DiffPath', 'GEPC': 'GEPC',
         'EigenScore': 'EigenScore', 'CFS': 'CFS (diag.)', 'CFS-kNN': 'CFS (kNN head)', 'RotCls4': 'Rotation pred.',
         'kNN (CT)': 'kNN (basic)', 'kNN tuned': 'kNN (tuned)'}
CITE = {'NLL': 'song2021score', 'IC-png': 'serra2020input', 'IC-j2k': 'serra2020input', 'TT': 'nalisnick2019typ', 'MSMA*': 'mahmood2021msma',
        'DiffPath': 'heng2024diffpath', 'GEPC': 'alexis2026gepc', 'EigenScore': 'shoushtari2026eigenscore', 'RotCls4': 'hendrycks2019ss',
        'CFS': 'rouzoumka2026cfs', 'CFS-kNN': 'rouzoumka2026cfs'}
PASSES = {'NLL': '128+128', 'IC-png': '128+128', 'IC-j2k': '128+128', 'TT': '128+128', 'MSMA*': '20', 'DiffPath': '10', 'GEPC': '14', 'EigenScore': '180',
          'RotCls4': '4$^{\\ddagger}$', 'kNN (CT)': '1', 'kNN tuned': '1', 'CFS': '1', 'CFS-kNN': '1'}
SEC1 = {json.loads(l)['method']: json.loads(l)['sec_per_img'] for l in open(f'{D}/bench_cpu_session.jsonl')}
SEC2 = {json.loads(l)['method']: json.loads(l)['sec_per_img'] for l in open(f'{D}/bench_cpu_session2.jsonl')}
SECROW = {'NLL': SEC1['nll'], 'IC-png': SEC1['ic'], 'IC-j2k': SEC1['ic'], 'TT': SEC1['nll'], 'MSMA*': SEC1['msma'], 'DiffPath': SEC1['diffpath'],
          'GEPC': SEC1['gepc'], 'EigenScore': SEC1['eigen'], 'CFS': SEC2['cfs'], 'CFS-kNN': SEC2['cfs_knn'], 'RotCls4': SEC2['rotcls4'],
          'kNN (CT)': SEC2['knn'], 'kNN tuned': SEC2['knn'], 'ours': SEC2['ours']}
def fsec(v): return f'{v:.0f}' if v >= 10 else (f'{v:.1f}' if v >= 1 else f'{v:.2f}')
def f1(v): return f'{v:.1f}'
J = json.load(open(f'{D}/ours_vs_all_final2.json')); JD = json.load(open(f'{D}/ours_vs_all_fresh.json'))
FP = json.load(open(f'{D}/fpr95_all_final2.json'))
def ids_mean(Jx, r): return float(np.mean([Jx['ids'][t]['rows'][r]['mean'] for t in TAGS]))
def vf_of(Jx, t, r=None): p = Jx['pairs'][t][VF[t]]; return p['ours'] if r is None else p['rows'][r]['auroc']
def wtl(Jx, r): v = Jx['rows'][r]; return f"{v['wins']}/{v['ties']}/{v['losses']}"
def pair(t, o, r=None): p = J['pairs'][t][o]; return p['ours'] if r is None else p['rows'][r]['auroc']
def is_nat(o): return not o.endswith('__vflip') and 'synth' not in o
NATP = [(t, o) for t in TAGS for o in J['pairs'][t] if is_nat(o)]
def wtl_nat_v(r):
    v = [J['pairs'][t][o]['rows'][r]['verdict'] for t, o in NATP if r in J['pairs'][t][o]['rows']]
    return v.count('win'), v.count('tie'), v.count('loss')
def wtl_nat(r): return '{}/{}/{}'.format(*wtl_nat_v(r))
M = {}
M['nPairs'] = str(sum(len(P) for P in J['pairs'].values()))
M['oursMean'] = f1(J['mean_ours']); M['tunedMean'] = f1(ids_mean(J, 'kNN tuned')); M['ctMean'] = f1(ids_mean(J, 'kNN (CT)'))
M['rotMean'] = f1(ids_mean(J, 'RotCls4')); M['cfsMean'] = f1(ids_mean(J, 'CFS')); M['cfsKnnMean'] = f1(ids_mean(J, 'CFS-kNN'))
best8 = max(EIGHT, key=lambda m: ids_mean(J, m)); M['bestEightName'] = LABEL[best8]; M['bestEightMean'] = f1(ids_mean(J, best8))
bestP = max(PRIOR, key=lambda m: ids_mean(J, m)); assert bestP == 'CFS-kNN'; M['bestPriorMean'] = f1(ids_mean(J, bestP))
M['priorGap'] = f1(J['mean_ours'] - ids_mean(J, bestP))
M['priorWinMin'] = str(min(J['rows'][m]['wins'] for m in PRIOR)); M['priorWinMax'] = str(max(J['rows'][m]['wins'] for m in PRIOR))
M['priorLossMax'] = str(max(J['rows'][m]['losses'] for m in PRIOR)); M['nPrior'] = str(len(PRIOR))
ROWS_ALL0 = ['kNN tuned', 'kNN (CT)'] + PRIOR
M['wtlTuned'] = wtl(J, 'kNN tuned'); M['wtlCfsKnn'] = wtl(J, 'CFS-kNN'); M['wtlRot'] = wtl(J, 'RotCls4')
M['nNatPairsMain'] = str(len(NATP))
M['priorWinMinNat'] = str(min(wtl_nat_v(m)[0] for m in PRIOR)); M['priorWinMaxNat'] = str(max(wtl_nat_v(m)[0] for m in PRIOR))
M['priorLossMaxNat'] = str(max(wtl_nat_v(m)[2] for m in PRIOR))
M['wtlTunedNat'] = wtl_nat('kNN tuned'); M['wtlKnnTNat'] = wtl_nat('Ours content'); M['wtlPoseNat'] = wtl_nat('Ours pose')
M['winKnnTNat'] = str(wtl_nat_v('Ours content')[0]); M['lossKnnTNat'] = str(wtl_nat_v('Ours content')[2])
M['gainKnnTNat'] = f1(np.mean([J['pairs'][t][o]['rows']['Ours content']['diff'] for t, o in NATP]))
M['knnTCten'] = f1(J['ids']['edmc10']['rows']['Ours content']['mean']); M['knnTIN'] = f1(J['ids']['in64']['rows']['Ours content']['mean'])
M['knnTMean'] = f1(ids_mean(J, 'Ours content'))
lostN = [(t, o) for t, o in NATP if any(J['pairs'][t][o]['rows'][r]['verdict'] == 'loss' for r in ROWS_ALL0 if r in J['pairs'][t][o]['rows'])]
assert sorted(lostN) == sorted([('l6svhn', 'celeba_test'), ('in64', 'in64_inaturalist'), ('edmc100', 'cifar10_test')]), lostN
M['nLostNat'] = str(len(lostN)); M['nNoLossNat'] = str(len(NATP) - len(lostN))
ROWS_ALL = ['kNN tuned', 'kNN (CT)'] + PRIOR
lost = [(t, o) for t, P in J['pairs'].items() for o, p in P.items() if any(p['rows'][r]['verdict'] == 'loss' for r in ROWS_ALL if r in p['rows'])]
M['nNoLoss'] = str(int(M['nPairs']) - len(lost)); M['nLost'] = str(len(lost))
assert sorted(lost) == sorted([('edmc10', 'cifar10_test__vflip'), ('edmc100', 'cifar100_test__vflip'), ('l6svhn', 'svhn_test__vflip'), ('in64', 'in64_imagenet_test__vflip'),
                               ('l6svhn', 'celeba_test'), ('in64', 'in64_inaturalist'), ('edmc100', 'cifar10_test')]), lost   # the text names these 7
vt = [vf_of(J, t, 'kNN tuned') for t in TAGS]; vo = [vf_of(J, t) for t in TAGS]
low4 = sorted(vt)[:4]; M['vfTunedFourRange'] = f'{min(low4):.0f}--{max(low4):.0f}'
M['vfOursAvg'] = f1(np.mean(vo)); M['vfTunedAvg'] = f1(np.mean(vt)); M['vfRotAvg'] = f1(np.mean([vf_of(J, t, 'RotCls4') for t in TAGS]))
M['vfPoseAvg'] = f1(np.mean([vf_of(J, t, 'Ours pose') for t in TAGS])); M['vfCfsKnnAvg'] = f1(np.mean([vf_of(J, t, 'CFS-kNN') for t in TAGS]))
M.update({'vfIN': f1(vf_of(J, 'in64')), 'vfINT': f1(vf_of(J, 'in64', 'kNN tuned')), 'vfSVHN': f1(vf_of(J, 'l6svhn')), 'vfSVHNPose': f1(vf_of(J, 'l6svhn', 'Ours pose')),
          'contentMean': f1(ids_mean(J, 'Ours content')), 'poseMean': f1(ids_mean(J, 'Ours pose')),
          'pChCten': f1(pair('edmc100', 'cifar10_test')), 'pChCtenT': f1(pair('edmc100', 'cifar10_test', 'kNN tuned')), 'poseChCten': f1(pair('edmc100', 'cifar10_test', 'Ours pose')),
          'pCtenCh': f1(pair('edmc10', 'cifar100_test')), 'pCtenChT': f1(pair('edmc10', 'cifar100_test', 'kNN tuned')), 'poseCtenCh': f1(pair('edmc10', 'cifar100_test', 'Ours pose')),
          'contentCtenCh': f1(pair('edmc10', 'cifar100_test', 'Ours content')),
          'pInInat': f1(pair('in64', 'in64_inaturalist')), 'pInInatNLL': f1(pair('in64', 'in64_inaturalist', 'NLL')),
          'oursCten': f1(J['ids']['edmc10']['ours']), 'tunedCten': f1(J['ids']['edmc10']['rows']['kNN tuned']['mean']),
          'oursIN': f1(J['ids']['in64']['ours']), 'tunedIN': f1(J['ids']['in64']['rows']['kNN tuned']['mean'])})
M['FPRours'] = f1(FP['mean']['Ours']['fpr95']); fb = min([r for r in FP['mean'] if r not in ('Ours', 'VL')], key=lambda r: FP['mean'][r]['fpr95'])
assert fb == 'kNN tuned'; M['FPRbest'] = f1(FP['mean'][fb]['fpr95'])
fpp = min(PRIOR, key=lambda r: FP['mean'][r]['fpr95']); M['FPRbestPriorName'] = LABEL[fpp]; M['FPRbestPrior'] = f1(FP['mean'][fpp]['fpr95'])
M.update({'cpuOurs': fsec(SECROW['ours']), 'cpuNll': fsec(SECROW['NLL']), 'cpuDiffPath': fsec(SECROW['DiffPath']), 'cpuCfs': fsec(SECROW['CFS']),
          'cpuEigen': fsec(SECROW['EigenScore']), 'cpuRot': fsec(SECROW['RotCls4']),
          'cpuRatioNll': f"{SECROW['NLL'] / SECROW['ours']:.0f}", 'cpuRatioDiffPath': f"{SECROW['DiffPath'] / SECROW['ours']:.0f}"})
EXPD = json.load(open(f'{D}/dR_s0.json')); ds = [float(EXPD[t]['d']) for t in TAGS]; M['dimRange'] = f'{min(ds):.0f}--{max(ds):.0f}'
ACC = {'edmc10': 95.5, 'l6svhn': 86.7, 'dpceleba': 100.0, 'edmc100': 88.2, 'in64': 77.3, 'fmnist': 99.7}   # logs/kx/headsR_<tag>.log (seed 0)
M['headAccRange'] = f'{min(ACC.values()):.0f}--{max(ACC.values()):.0f}'; M['leakAccIN'] = '91.6'; M['cleanAccIN'] = '77.3'
M['devOurs'] = f1(JD['mean_ours']); M['devTuned'] = f1(ids_mean(JD, 'kNN tuned'))
AB = {t: json.load(open(f'{D}/ablate2_{t}.json')) for t in TAGS}
for k, c in (('ours', 'A'), ('content only', 'B'), ('orientation only', 'C'), ('no temperature', 'D'), ('Fisher instead of sum', 'E'), ('orbit bank', 'F'),
             ('head with rotated noise', 'G'), ('four views (rotated-noise test views)', 'H')):
    M[f'abNat{c}'] = f1(np.mean([AB[t][k]['nat'] for t in TAGS])); M[f'abVf{c}'] = f1(np.mean([AB[t][k]['vflip'] for t in TAGS])); M[f'abSel{c}'] = f1(np.mean([AB[t][k]['sel'] for t in TAGS]))
M['abInVfLeak'] = f1(AB['in64']['head with rotated noise']['vflip']); M['abInVfNoT'] = f1(AB['in64']['no temperature']['vflip'])
M['abCtenOurs'] = f1(AB['edmc10']['ours']['nat']); M['abCtenOrbit'] = f1(AB['edmc10']['orbit bank']['nat'])
# ---- OpenOOD v1.5 ImageNet-1K at 64x64
O = json.load(open(f'{D}/ood1k_ana.json')); C = O['common']
def best_prior(key): return max(['CFS', 'CFS kNN', 'RotCls4', 'NLL', 'IC-png', 'IC-j2k', 'TT', 'MSMA*', 'DiffPath', 'GEPC', 'EigenScore'], key=lambda r: C[r][key])
M.update({'inNear': f1(C['Ours']['near']), 'inFar': f1(C['Ours']['far']), 'inNearPrior': f1(C[best_prior('near')]['near']), 'inFarPrior': f1(C[best_prior('far')]['far'])})
assert best_prior('near') == 'CFS' and best_prior('far') == 'CFS'
M.update({'inSSB': f1(C['Ours']['in64_ssb_hard']['auroc']), 'inNINCO': f1(C['Ours']['in64_ninco']['auroc']), 'inInat': f1(C['Ours']['in64_inaturalist']['auroc']),
          'inTex': f1(C['Ours']['in64_textures']['auroc']), 'inOIO': f1(C['Ours']['in64_openimage_o']['auroc']),
          'inInatNLL': f1(C['NLL']['in64_inaturalist']['auroc']), 'inInatTT': f1(C['TT']['in64_inaturalist']['auroc']),
          'inVtwo': f1(C['Ours']['in64_imagenet_v2']['auroc']), 'inC': f1(C['Ours']['in64_imagenet_c']['auroc']), 'inR': f1(C['Ours']['in64_imagenet_r']['auroc']),
          'inFSnear': f1(C['Ours']['FS-near']), 'inFSfar': f1(C['Ours']['FS-far']), 'inFSnearNLL': f1(C['NLL']['FS-near']), 'inFSfarNLL': f1(C['NLL']['FS-far']),
          'inCNLL': f1(C['NLL']['in64_imagenet_c']['auroc']), 'inRNLL': f1(C['NLL']['in64_imagenet_r']['auroc']),
          'inNearFPR': f1(C['Ours']['near_fpr95']), 'inFarFPR': f1(C['Ours']['far_fpr95']),
          'inNearKnnT': f1(C['kNN term']['near']), 'inFarKnnT': f1(C['kNN term']['far']), 'inNearPose': f1(C['Orientation term']['near']), 'inFarPose': f1(C['Orientation term']['far'])})
M['inNearFull'] = f1(O['full']['Ours']['near']); M['inFarFull'] = f1(O['full']['Ours']['far'])
sev = O['imagenet_c']['Ours']['severity']; M['inCsevOne'] = f1(sev['1']); M['inCsevFive'] = f1(sev['5'])
pw = {r: O['boot'][r]['wtl'] for r in ['CFS', 'CFS kNN', 'RotCls4', 'NLL', 'IC-png', 'IC-j2k', 'TT', 'MSMA*', 'DiffPath', 'GEPC', 'EigenScore']}
M['inPriorWinMin'] = str(min(v[0] for v in pw.values())); M['inPriorLossMax'] = str(max(v[2] for v in pw.values()))
M['inLossRows'] = ', '.join(sorted(LABEL.get(r, r) for r, v in pw.items() if v[2] > 0))
M['inNFull'] = f"{O['n_full']['in64_imagenet_test']:,}".replace(',', '{,}')
# ---- naive fusions (E2)
F = json.load(open(f'{D}/fusion_final2.json'))
for k, c in (('kNN+RotCls4 (z-sum)', 'RZ'), ('kNN+RotCls4 (Fisher)', 'RF'), ('kNN+Orient. (z-sum)', 'OZ'), ('kNN+Orient. (Fisher)', 'OF'), ('Ours, d = 1', 'DO')):
    M[f'fu{c}'] = f1(F['mean'][k]); M[f'fu{c}vf'] = f1(F['vflip_mean'][k]); w = F['wtl_nat'][k]; M[f'fu{c}wtl'] = f"{w['win']}/{w['tie']}/{w['loss']}"
assert abs(F['mean']['Ours'] - J['mean_ours']) < 1e-6
M['nNatPairs'] = str(sum(F['wtl_nat']['kNN term'].values()))
# ---- ImageNet 256x256 (ADM): in256_ana.py.  SSB-hard and ImageNet-C are distributed at 224 px (short side) and would be upsampled to 256,
#   unlike the ID images (median short side 375) and the other sets, so the 256 comparison uses NINCO (near) and the three far sets.
H = json.load(open(f'{D}/in256_ana.json'))
HC, HA = H['common'], H['all']
P256 = ['CFS', 'CFS kNN', 'MSMA*', 'DiffPath', 'GEPC']
CLEAN = ['ninco', 'inaturalist', 'textures', 'openimage_o']; FAR3 = ['inaturalist', 'textures', 'openimage_o']
def h_far(r): return float(np.mean([HC[r][s]['auroc'] for s in FAR3]))
M.update({'hNINCO': f1(HC['Ours']['ninco']['auroc']), 'hFar': f1(h_far('Ours')), 'hVf': f1(HC['Ours']['imagenet_test__vflip']['auroc']),
          'hInat': f1(HC['Ours']['inaturalist']['auroc']), 'hTex': f1(HC['Ours']['textures']['auroc']), 'hOIO': f1(HC['Ours']['openimage_o']['auroc']),
          'hNINCOKnnT': f1(HC['kNN term']['ninco']['auroc']), 'hVfKnnT': f1(HC['kNN term']['imagenet_test__vflip']['auroc']),
          'hVfPose': f1(HC['Orientation term']['imagenet_test__vflip']['auroc']), 'hNcom': str(H['n_common']['imagenet_test']),
          'hSSB': f1(HC['Ours']['ssb_hard']['auroc']), 'hSSBCfs': f1(HC['CFS']['ssb_hard']['auroc']), 'hSSBKnnB': f1(HC['kNN basic']['ssb_hard']['auroc'])})
bn = max(P256, key=lambda r: HC[r]['ninco']['auroc']); M['hNINCOPrior'] = f1(HC[bn]['ninco']['auroc']); M['hNINCOPriorName'] = LABEL.get({'CFS kNN': 'CFS-kNN'}.get(bn, bn), bn)
bf = max(P256, key=h_far); M['hFarPrior'] = f1(h_far(bf)); M['hFarPriorName'] = LABEL.get({'CFS kNN': 'CFS-kNN'}.get(bf, bf), bf)
M['hFarVerdictCfs'] = H['boot']['CFS']['far']['verdict']
for r, c in (('CFS', 'Cfs'), ('MSMA*', 'Msma')):
    M[f'hTex{c}'] = f1(HC[r]['textures']['auroc']); M[f'hOIO{c}'] = f1(HC[r]['openimage_o']['auroc'])
M['hVfOtherMax'] = f1(max(HC[r]['imagenet_test__vflip']['auroc'] for r in P256 + ['kNN basic']))
M.update({'hVtwo': f1(HA['Ours']['imagenet_v2']['auroc']), 'hC': f1(HA['Ours']['imagenet_c']['auroc']), 'hR': f1(HA['Ours']['imagenet_r']['auroc'])})
M['hSeeds'] = str(len(H['seeds']['Ours'])); M['hCfsSeeds'] = str(len(H['seeds']['CFS']))
def wtl4(r): v = [H['boot'][r][s]['verdict'] for s in CLEAN]; return f"{v.count('win')}/{v.count('tie')}/{v.count('loss')}"
M['hTempRange'] = f"{min(t['mlp'] for t in H['temp'].values()):.2f}--{max(t['mlp'] for t in H['temp'].values()):.2f}"
# ---- EuroSAT (E3): e3_ana.py, optional until the runs finish
E = json.load(open(f'{D}/e3_ana.json')) if os.path.exists(f'{D}/e3_ana.json') else None
if E:
    ER = E['rows']
    M.update({'eOurs': f1(ER['Ours']['mean']), 'eKnnT': f1(ER['kNN term']['mean']), 'ePose': f1(ER['Orientation term']['mean']), 'eKnnB': f1(ER['kNN basic']['mean']),
              'eVf': f1(ER['Ours']['vflip']), 'eRot': f1(ER['Ours']['rot90']), 'eVfKnnT': f1(ER['kNN term']['vflip']), 'eRotKnnT': f1(ER['kNN term']['rot90']),
              'eVfPose': f1(ER['Orientation term']['vflip']), 'eRotPose': f1(ER['Orientation term']['rot90']),
              'eWtlKnnT': '{}/{}/{}'.format(*E['boot']['kNN term']['wtl']), 'eWtlKnnB': '{}/{}/{}'.format(*E['boot']['kNN basic']['wtl']),
              'eMeanVerdictKnnT': E['boot']['kNN term']['mean']['verdict']})
    accs = [h['acc'] for h in E['heads'].values()]; Ts = [h['T'] for h in E['heads'].values()]
    M['eAcc'] = f'{min(accs):.0f}--{max(accs):.0f}' if round(min(accs)) != round(max(accs)) else f'{min(accs):.0f}'
    M['eT'] = f'{min(Ts):.2f}--{max(Ts):.2f}' if round(min(Ts), 2) != round(max(Ts), 2) else f'{min(Ts):.2f}'
assert all(not any(ch.isdigit() for ch in k) for k in M)
with open(f'{OUT}/numbers.tex', 'w') as fo:
    fo.write('% written by gen/mk_ijcai.py\n')
    for k, v in M.items(): fo.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')


# ---- Table 1 (6 ID tiers, final split)
def marks(vals):
    """per column: best -> light blue + bold, second -> underline (displayed values; ties share the mark)"""
    r = [round(v, 1) for v in vals]; u = sorted(set(r), reverse=True)
    return [('b' if x == u[0] else ('s' if len(u) > 1 and x == u[1] else '')) for x in r]
def cell(v, m): return {'b': f'\\cellcolor{{firstblue}}\\textbf{{{f1(v)}}}', 's': f'\\underline{{{f1(v)}}}', '': f1(v)}[m]
rows = PRIOR + ['kNN (CT)', 'kNN tuned']
cols = {r: [J['ids'][t]['rows'][r]['mean'] for t in TAGS] + [ids_mean(J, r), float(np.mean([vf_of(J, t, r) for t in TAGS]))] for r in rows}
cols['ours'] = [J['ids'][t]['ours'] for t in TAGS] + [J['mean_ours'], float(np.mean(vo))]
order = rows + ['ours']; mk = {r: [] for r in order}
for j in range(8):
    mm = marks([cols[r][j] for r in order])
    for r, m in zip(order, mm): mk[r].append(m)
L = ['% written by gen/mk_ijcai.py']
for r in rows:
    if r == 'kNN (CT)': L.append('\\midrule\\multicolumn{12}{@{}l}{\\emph{kNN rule~\\cite{sun2022knn} on our diffusion features}}\\\\')
    lab = LABEL[r]                                            # citations are in Section 4 (the named style is too wide for the table)
    L.append(lab + ' & ' + ' & '.join(cell(v, m) for v, m in zip(cols[r], mk[r])) + f" & {wtl_nat(r)} & {PASSES[r]} & {fsec(SECROW[r])}\\\\")
    if r == 'RotCls4' and not os.path.exists(f'{HERE}/data/csi_ana.json'):   # CSI (PROTOCOL_CSI.md): row filled when its runs finish
        L.append('CSI & \\multicolumn{11}{l}{\\pending{1{,}000-epoch training on the five 32$\\times$32 ID datasets still running}}\\\\')
L.append('\\midrule')
L.append('\\textbf{Upright kNN (ours)} & ' + ' & '.join(cell(v, m) for v, m in zip(cols['ours'], mk['ours'])) + f" & -- & 1 & {fsec(SECROW['ours'])}\\\\")
for key, lab in (('Ours content', '\\quad kNN term only'), ('Ours pose', '\\quad orientation term only')):
    vals = [J['ids'][t]['rows'][key]['mean'] for t in TAGS] + [ids_mean(J, key), float(np.mean([vf_of(J, t, key) for t in TAGS]))]
    L.append(f'{lab} & ' + ' & '.join(f1(v) for v in vals) + f" & {wtl_nat(key)} & 1 & --\\\\")
open(f'{OUT}/tab_main.tex', 'w').write('\n'.join(L) + '\n')


# ---- Table 2 (ImageNet-1K, OpenOOD v1.5): 64x64 block (+ 256x256 block when available)
def irow(lab, Mx, key_cs, mkd):
    v = [Mx['near'], Mx['far']]
    cs = [Mx[k]['auroc'] for k in key_cs] if key_cs else None
    fs = [Mx['FS-near'], Mx['FS-far']] if 'FS-near' in Mx else None
    out = [cell(v[0], mkd[0]), cell(v[1], mkd[1])]
    out += [f1(x) for x in cs] if cs else ['--'] * 3
    out += [cell(fs[0], mkd[2]), cell(fs[1], mkd[3])] if fs else ['--', '--']
    return lab + ' & ' + ' & '.join(out) + '\\\\'
R64 = ['Ours', 'kNN term', 'Orientation term', 'kNN basic', 'CFS', 'CFS kNN', 'RotCls4', 'NLL', 'IC-png', 'IC-j2k', 'TT', 'MSMA*', 'DiffPath', 'GEPC', 'EigenScore']
LAB2 = {'Ours': '\\textbf{Upright kNN}', 'kNN term': '\\quad kNN term', 'Orientation term': '\\quad orient.\\ term', 'kNN basic': 'kNN (basic)',
        'CFS': 'CFS (diag.)', 'CFS kNN': 'CFS (kNN)', 'RotCls4': 'Rotation pred.', 'NLL': 'NLL', 'IC-png': 'IC (PNG)', 'IC-j2k': 'IC (J2K)', 'TT': 'TT',
        'MSMA*': 'MSMA$^{*}$', 'DiffPath': 'DiffPath', 'GEPC': 'GEPC', 'EigenScore': 'EigenScore'}
CS64 = ['in64_imagenet_v2', 'in64_imagenet_c', 'in64_imagenet_r']
ABL = ('kNN term', 'Orientation term')                          # the two terms of Ours: shown without marks
def block(Rs, get, cs_of):
    keys = ['near', 'far', 'FS-near', 'FS-far']; mkd = {r: [] for r in Rs}
    for k in keys:
        rr = [r for r in Rs if k in get(r) and r not in ABL]
        mm = marks([get(r)[k] for r in rr]); d = dict(zip(rr, mm))
        for r in Rs: mkd[r].append(d.get(r, ''))
    return [irow(LAB2[r], get(r), cs_of(r), mkd[r]) for r in Rs]
L = ['% written by gen/mk_ijcai.py']
L += block(R64, lambda r: C[r], lambda r: CS64)
open(f'{OUT}/tab_imagenet.tex', 'w').write('\n'.join(L) + '\n')


# ---- Table 3 (ImageNet 256x256, ADM): NINCO and the far sets, upside-down ID images; W/T/L of ours over the four sets
R256 = ['Ours', 'kNN term', 'Orientation term', 'kNN basic', 'CFS', 'CFS kNN', 'MSMA*', 'DiffPath', 'GEPC']
cols256 = {r: [HC[r]['ninco']['auroc']] + [HC[r][s]['auroc'] for s in FAR3] + [h_far(r), HC[r]['imagenet_test__vflip']['auroc']] for r in R256}
mk3 = {r: [] for r in R256}
for j in range(6):
    rr = [r for r in R256 if r not in ABL]; mm = dict(zip(rr, marks([cols256[r][j] for r in rr])))
    for r in R256: mk3[r].append(mm.get(r, ''))
L = ['% written by gen/mk_ijcai.py']
for r in R256:
    w = '--' if r == 'Ours' else wtl4(r)
    L.append(LAB2[r] + ' & ' + ' & '.join(cell(v, m) for v, m in zip(cols256[r], mk3[r])) + f' & {w}\\\\')
    if r in ('Orientation term', 'kNN basic'): L.append('\\midrule')
open(f'{OUT}/tab_in256.tex', 'w').write('\n'.join(L) + '\n')


# ---- Table 4 (naive fusions, final split)
FR = [('Ours', '\\textbf{Upright kNN} (sum, Eq.~\\ref{eq:score})'), ('kNN term', 'kNN term only'), ('Orientation term', 'orientation term only'),
      ('RotCls4', 'rotation classifier (ResNet-18)'), ('kNN+RotCls4 (z-sum)', 'kNN + rot.\\ classifier, $z$-sum'),
      ('kNN+RotCls4 (Fisher)', 'kNN + rot.\\ classifier, Fisher'), ('kNN+Orient. (z-sum)', 'kNN + orient.\\ term, $z$-sum'),
      ('kNN+Orient. (Fisher)', 'kNN + orient.\\ term, Fisher'), ('Ours, d = 1', 'Upright kNN with $d=1$')]
L = ['% written by gen/mk_ijcai.py']
mm = dict(zip([k for k, _ in FR], marks([F['mean'][k] for k, _ in FR])))
for k, lab in FR:
    w = '--' if k == 'Ours' else '{win}/{tie}/{loss}'.format(**F['wtl_nat'][k])
    L.append(f"{lab} & {cell(F['mean'][k], mm[k])} & {f1(F['vflip_mean'][k])} & {w}\\\\")
    if k in ('Ours', 'RotCls4'): L.append('\\midrule')
open(f'{OUT}/tab_fusion.tex', 'w').write('\n'.join(L) + '\n')
print(json.dumps(M, indent=0))
