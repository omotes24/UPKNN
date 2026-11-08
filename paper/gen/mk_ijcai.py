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
NEWJ = json.load(open(f'{D}/newrows_final2.json'))                 # DDPM-OOD (Graham), LMD, likelihood + LID (Kamkari): newrows_final2.py
NEWROWS = [r for r in ('Graham', 'LMD', 'Kamkari') if r in NEWJ['rows']]
CSI = json.load(open(f'{D}/csi_ana.json')) if os.path.exists(f'{D}/csi_ana.json') else None   # CSI on the five 32x32 IDs: csi_ana.py
PRIOR = EIGHT + NEWROWS + ['CFS', 'CFS-kNN', 'RotCls4']
IDN0 = {'edmc10': 'CIFAR-10', 'l6svhn': 'SVHN', 'dpceleba': 'CelebA', 'edmc100': 'CIFAR-100', 'in64': 'ImageNet-64', 'fmnist': 'FashionMNIST'}
SETN0 = {'cifar10_test': 'CIFAR-10', 'cifar100_test': 'CIFAR-100', 'celeba_test': 'CelebA', 'svhn_test': 'SVHN', 'textures': 'Textures', 'tin_val': 'Tiny-ImageNet', 'lsun_fix': 'LSUN-FIX',
         'imagenet_fix': 'ImageNet-FIX', 'imagenet32_val': 'ImageNet32', 'places365_val': 'Places365'}
LABEL = {'NLL': 'NLL', 'IC-png': 'IC (PNG)', 'IC-j2k': 'IC (JPEG2K)', 'TT': 'TT', 'MSMA*': 'MSMA$^{*}$', 'DiffPath': 'DiffPath', 'GEPC': 'GEPC',
         'EigenScore': 'EigenScore', 'CFS': 'CFS (diag.)', 'CFS-kNN': 'CFS (kNN)', 'RotCls4': 'Rotation pred.',
         'Graham': 'DDPM-OOD', 'LMD': 'LMD', 'Kamkari': 'LID+NLL', 'CSI': 'CSI',
         'kNN (CT)': 'kNN (basic)', 'kNN tuned': 'kNN (tuned)'}
CITE = {'NLL': 'song2021score', 'IC-png': 'serra2020input', 'IC-j2k': 'serra2020input', 'TT': 'nalisnick2019typ', 'MSMA*': 'mahmood2021msma',
        'DiffPath': 'heng2024diffpath', 'GEPC': 'alexis2026gepc', 'EigenScore': 'shoushtari2026eigenscore', 'RotCls4': 'hendrycks2019ss',
        'CFS': 'rouzoumka2026cfs', 'CFS-kNN': 'rouzoumka2026cfs'}
PASSES = {'NLL': '256', 'IC-png': '256', 'IC-j2k': '256', 'TT': '256', 'MSMA*': '20', 'DiffPath': '10', 'GEPC': '14', 'EigenScore': '180',
          'Graham': '2{,}475', 'LMD': '10{,}000', 'Kamkari': '12{,}544', 'CSI': '40$^{\\ddagger}$',
          'RotCls4': '4$^{\\ddagger}$', 'kNN (CT)': '1', 'kNN tuned': '1', 'CFS': '1', 'CFS-kNN': '1'}
SEC1 = {json.loads(l)['method']: json.loads(l)['sec_per_img'] for l in open(f'{D}/bench_cpu_session.jsonl')}
SEC2 = {json.loads(l)['method']: json.loads(l)['sec_per_img'] for l in open(f'{D}/bench_cpu_session2.jsonl')}
SECROW = {'NLL': SEC1['nll'], 'IC-png': SEC1['ic'], 'IC-j2k': SEC1['ic'], 'TT': SEC1['nll'], 'MSMA*': SEC1['msma'], 'DiffPath': SEC1['diffpath'],
          'GEPC': SEC1['gepc'], 'EigenScore': SEC1['eigen'], 'CFS': SEC2['cfs'], 'CFS-kNN': SEC2['cfs_knn'], 'RotCls4': SEC2['rotcls4'],
          'kNN (CT)': SEC2['knn'], 'kNN tuned': SEC2['knn'], 'ours': SEC2['ours'], 'Graham': None, 'LMD': None, 'Kamkari': None, 'CSI': None}
def fsec(v): return '--' if v is None else (f'{v:.0f}' if v >= 10 else (f'{v:.1f}' if v >= 1 else f'{v:.2f}'))
def f1(v): return f'{v:.1f}'
J = json.load(open(f'{D}/ours_vs_all_final2.json')); JD = json.load(open(f'{D}/ours_vs_all_fresh.json'))
for t in J['pairs']:                                                    # merge the new comparison rows (same images, same bootstrap draws)
    for o, p in J['pairs'][t].items():
        for r in NEWROWS:
            if r in NEWJ['pairs'][t][o]['rows']: p['rows'][r] = NEWJ['pairs'][t][o]['rows'][r]
    for r in NEWROWS:
        if r in NEWJ['ids'][t]['rows']: J['ids'][t]['rows'][r] = NEWJ['ids'][t]['rows'][r]
for r in NEWROWS: J['rows'][r] = dict(wins=NEWJ['rows'][r]['wins'], ties=NEWJ['rows'][r]['ties'], losses=NEWJ['rows'][r]['losses'])
NEWTIERS = {r: NEWJ['rows'][r]['tiers'] for r in NEWROWS}            # tiers with complete runs (all six when the runs are finished)
FP = json.load(open(f'{D}/fpr95_all_final2.json'))
def ids_mean(Jx, r): return float(np.mean([Jx['ids'][t]['rows'][r]['mean'] for t in TAGS if r in Jx['ids'][t]['rows']]))
NEAR4 = [('edmc10', 'cifar100_test'), ('edmc100', 'cifar10_test'), ('in64', 'in64_ssb_hard'), ('in64', 'in64_ninco')]   # the near-OOD pairs
def near_mean(r=None):
    v = [(J['pairs'][t][o]['ours'] if r is None else J['pairs'][t][o]['rows'][r]['auroc']) for t, o in NEAR4 if r is None or r in J['pairs'][t][o]['rows']]
    return float(np.mean(v)) if len(v) == 4 else None
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
STAT = [m for m in EIGHT + NEWROWS if all(m in J['ids'][t]['rows'] for t in TAGS)]
bestS = max(STAT, key=lambda m: ids_mean(J, m)); M['bestStatName'] = LABEL[bestS]; M['bestStatMean'] = f1(ids_mean(J, bestS)); M['nStat'] = str(len(STAT))
bestP = max([m for m in PRIOR if all(m in J['ids'][t]['rows'] for t in TAGS)], key=lambda m: ids_mean(J, m)); assert bestP == 'CFS-kNN'; M['bestPriorMean'] = f1(ids_mean(J, bestP))
M['nPriorComplete'] = str(len([m for m in PRIOR if all(m in J['ids'][t]['rows'] for t in TAGS)]))
M['priorGap'] = f1(J['mean_ours'] - ids_mean(J, bestP))
M['featGap'] = f1(ids_mean(J, 'kNN tuned') - ids_mean(J, bestP)); M['frameGapTuned'] = f1(J['mean_ours'] - ids_mean(J, 'kNN tuned'))   # of the gap to the best prior detector: features (tuned kNN) and the frame term
COMPLETE = [m for m in PRIOR if all(m in J['ids'][t]['rows'] for t in TAGS)]        # rows run on all six ID datasets (win counts are comparable)
M['priorWinMin'] = str(min(J['rows'][m]['wins'] for m in COMPLETE)); M['priorWinMax'] = str(max(J['rows'][m]['wins'] for m in COMPLETE))
M['priorLossMax'] = str(max(J['rows'][m]['losses'] for m in PRIOR)); M['nPrior'] = str(len(PRIOR)); M['nPriorModel'] = str(len(PRIOR) - 1)   # all but the rotation classifier use the diffusion model
NUMWORD = {8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen'}
M['nPriorModelWord'] = NUMWORD[len(PRIOR) - 1]; M['nPriorWord'] = NUMWORD[len(PRIOR)]
ROWS_ALL0 = ['kNN tuned', 'kNN (CT)'] + PRIOR
M['wtlTuned'] = wtl(J, 'kNN tuned'); M['wtlCfsKnn'] = wtl(J, 'CFS-kNN'); M['wtlRot'] = wtl(J, 'RotCls4')
M['nNatPairsMain'] = str(len(NATP))
# near-OOD means (CIFAR-10 <-> CIFAR-100, SSB-hard, NINCO) and wins / losses of at least one AUROC point
M['nearOurs'] = f1(near_mean()); M['nearKnnT'] = f1(near_mean('Ours content')); M['nearTuned'] = f1(near_mean('kNN tuned')); M['nearPose'] = f1(near_mean('Ours pose'))
bestNear = max([m for m in PRIOR if near_mean(m) is not None], key=near_mean); M['nearBestPrior'] = f1(near_mean(bestNear)); M['nearBestPriorName'] = LABEL[bestNear]
def big_wtl(r, thr=1.0):
    w = l = 0
    for t, o in NATP:
        if r not in J['pairs'][t][o]['rows']: continue
        v = J['pairs'][t][o]['rows'][r]
        if v['verdict'] == 'win' and v['diff'] >= thr: w += 1
        if v['verdict'] == 'loss' and v['diff'] <= -thr: l += 1
    return w, l
M['bigWinKnnT'], M['bigLossKnnT'] = map(str, big_wtl('Ours content')); M['bigWinTuned'], M['bigLossTuned'] = map(str, big_wtl('kNN tuned'))
M['bigWinCfsKnn'], M['bigLossCfsKnn'] = map(str, big_wtl('CFS-kNN'))
M['bigWinPriorMin'] = str(min(big_wtl(m)[0] for m in PRIOR if all(m in J['ids'][t]['rows'] for t in TAGS)))
sdm = []
for t in TAGS:
    nat_sets = [o for o in J['pairs'][t] if is_nat(o)]
    sdm.append(float(np.std(np.array([[J['pairs'][t][o]['ours_seeds'][k] for o in nat_sets] for k in range(3)]).mean(1))))
M['seedSdMax'] = f'{max(sdm):.2f}'
# weight sweep (development images): weight_sweep.py
WS = json.load(open(f'{D}/weight_sweep.json'))['fresh']; wi = {w: i for i, w in enumerate(WS['w'])}
M['swZero'] = f1(WS['mean_nat'][wi[0]]); M['swOne'] = f1(WS['mean_nat'][wi[1]]); M['swBest'] = f1(max(WS['mean_nat'])); M['swBestW'] = str(WS['argmax_w'])
plateau = [w for w, v in zip(WS['w'], WS['mean_nat']) if v >= max(WS['mean_nat']) - 0.1]; M['swLo'] = str(min(plateau)); M['swHi'] = str(max(plateau))
M['swFour'] = f1(WS['mean_nat'][wi[4]]); M['swHalf'] = f1(WS['mean_nat'][wi[0.5]]); M['swTwo'] = f1(WS['mean_nat'][wi[2]])
M['swDprimeGap'] = f'{max(WS["mean_nat"]) - min(WS["mean_nat"][wi[0.75]], WS["mean_nat"][wi[1.25]]):.2f}'   # the weights implied by d estimated with 5 or 50 neighbours
# the three hardest near-OOD pairs: the sum against the better of its two terms (per-pair complementarity is not guaranteed)
for (t, o), c in zip([('edmc10', 'cifar100_test'), ('edmc100', 'cifar10_test'), ('in64', 'in64_ssb_hard')], ('CtenCh', 'ChCten', 'InSSB')):
    bt = max(('Ours content', 'Ours pose'), key=lambda r: pair(t, o, r)); M[f'p{c}Sum'] = f1(pair(t, o)); M[f'p{c}BestTerm'] = f1(pair(t, o, bt))
    M[f'p{c}BestTermName'] = {'Ours content': 'kNN term', 'Ours pose': 'orientation term'}[bt]
# mirror test (development images, seed 0): horizontal flip AUROC of the kNN term of Upright kNN -- the premise of Proposition 2 holds where it is near 50
MIR = {l.split('&')[0].strip(): [c.strip() for c in l.split('&')] for l in open(f'{D}/tab_mirror_rows.tex') if l.strip() and not l.startswith('%')}
hk = {k: float(v[11]) for k, v in MIR.items()}; M['mirHknnSVHN'] = f1(hk['SVHN']); M['mirHknnFM'] = f1(hk['FashionMNIST'])
M['mirHknnOtherMax'] = f1(max(v for k, v in hk.items() if k not in ('SVHN', 'FashionMNIST'))); M['mirVknnFM'] = f1(float(MIR['FashionMNIST'][10]))
# CSI (five 32x32 ID datasets)
if CSI:
    M['csiMean'] = f1(CSI['mean_nat']['csi']); M['csiOursMean'] = f1(CSI['mean_nat']['ours']); w = CSI['wtl_nat']; M['csiWtl'] = f"{w['wins']}/{w['ties']}/{w['losses']}"; M['csiNatPairs'] = str(w['pairs'])
    M['csiCtenCh'] = f1(CSI['pairs']['edmc10']['cifar100_test']['csi']); M['csiChCten'] = f1(CSI['pairs']['edmc100']['cifar10_test']['csi'])
    M['csiRepro'] = f1(CSI['repro_c10_c100']); M['csiPublished'] = f1(CSI['published_c10_c100'])
    M['csiChMean'] = f1(CSI['ids']['edmc100']['csi_nat']); M['csiChCeleb'] = f1(CSI['pairs']['edmc100']['celeba_test']['csi']); M['csiChNoise'] = f1(CSI['pairs']['edmc100']['synth_noise']['csi'])
    M['csiVfMean'] = f1(np.mean([v['csi'] for v in CSI['upside_down'].values()])); M['csiOursVfMean'] = f1(np.mean([v['ours'] for v in CSI['upside_down'].values()]))
    hrs = [c['hours'] for c in CSI['cost'].values()]; M['csiHoursMin'] = f'{min(hrs):.0f}'; M['csiHoursMax'] = f'{max(hrs):.0f}'
    lossp = [(t, o) for t in CSI['pairs'] for o, p in CSI['pairs'][t].items() if p['verdict'] == 'loss' and p['kind'] == 'natural']
    M['csiLossPairs'] = str(len(lossp))
    CSN = {'cifar100_test': 'CIFAR-100', 'cifar10_test': 'CIFAR-10', 'omniglot': 'Omniglot', 'g32_omniglot': 'Omniglot', 'g32_omniglot_eval': 'Omniglot'}
    M['csiLossList'] = ', '.join(f"{IDN0[t]}$\\to${CSN.get(o, SETN0.get(o, o))}" for t, o in lossp)
    M['csiLsunFix'] = f1(CSI['pairs']['edmc10']['lsun_fix']['csi']); M['csiInFix'] = f1(CSI['pairs']['edmc10']['imagenet_fix']['csi'])
    M['csiChVf'] = f1(CSI['upside_down']['edmc100']['csi'])
# the three further prior methods
KAM_ALL = ['edmc10', 'l6svhn', 'dpceleba', 'edmc100', 'fmnist']     # LID+NLL: the public configurations cover the five 32x32 ID datasets (none for ImageNet-64)
for r, c in (('Graham', 'Gr'), ('LMD', 'Lmd'), ('Kamkari', 'Kam')):
    if r in NEWROWS:
        M[f'new{c}Mean'] = f1(NEWJ['rows'][r]['mean_nat']); v = NEWJ['rows'][r]; M[f'new{c}Wtl'] = f"{v['nat_wins']}/{v['nat_ties']}/{v['nat_losses']}"; M[f'new{c}Tiers'] = str(len(v['tiers']))
        M[f'new{c}OursMean'] = f1(np.mean([J['ids'][t]['ours'] for t in v['tiers']])); M[f'new{c}Pairs'] = str(v['nat_pairs'])   # Upright kNN on the same ID datasets
        M[f'new{c}TierList'] = ', '.join(IDN0[t] for t in TAGS if t in v['tiers'])
# LID+NLL: its scope (the five 32x32 datasets once finished) and a pending note while runs are missing; both empty themselves out when the runs are complete
kam_done = NEWTIERS.get('Kamkari', []); kam_missing = [t for t in KAM_ALL if t not in kam_done]
for k in ('newKamMean', 'newKamOursMean', 'newKamWtl', 'newKamPairs', 'newKamTiers', 'newKamTierList'): M.setdefault(k, '--')   # before any LID+NLL run is complete
M['newKamScope'] = 'the five 32$\\times$32 datasets' if not kam_missing else ((', '.join(IDN0[t] for t in TAGS if t in kam_done) if kam_done else 'no ID dataset') + ' so far')
M['pendingKam'] = '' if not kam_missing else ' \\pending{LID+NLL: ' + ', '.join(IDN0[t] for t in kam_missing) + ' still running}'
# E4: chest X-ray, histology, generators trained with rotations (e4_ana.py)
E4 = json.load(open(f'{D}/e4_ana.json'))
def e4r(tag, row, key): return E4[tag]['rows'][row][key]
def e4t(tag, row, tf): return E4[tag]['rows'][row]['per_set'][f"{ {'chest': 'g32_chest_test', 'path': 'path_test'}[tag] }__{tf}"]
M.update({'xrAcc': f1(100 * np.mean([h['acc'] for h in E4['chest']['heads'].values()]) / 100), 'xrRotPose': f1(min(e4t('chest', 'Orientation term', t) for t in ('rot90', 'rot180', 'vflip'))),
          'xrRotKnnT': f1(min(e4t('chest', 'kNN term', t) for t in ('rot90', 'rot180', 'vflip'))), 'xrHflip': f1(e4t('chest', 'Ours', 'hflip')),
          'xrMod': f1(e4r('chest', 'Ours', 'modality')), 'xrModKnnT': f1(e4r('chest', 'kNN term', 'modality')), 'xrNat': f1(e4r('chest', 'Ours', 'natural')),
          'xrModNLL': f1(e4r('chest', 'NLL', 'modality')), 'xrRotNLL': f1(min(e4t('chest', 'NLL', t) for t in ('rot90', 'rot180', 'vflip'))),
          'xrRotCfs': f1(min(e4t('chest', 'CFS', t) for t in ('rot90', 'rot180', 'vflip'))),
          'phAcc': f1(np.mean([h['acc'] for h in E4['path']['heads'].values()])), 'phMod': f1(e4r('path', 'Ours', 'modality')), 'phModKnnT': f1(e4r('path', 'kNN term', 'modality')),
          'phNat': f1(e4r('path', 'Ours', 'natural')), 'phNatKnnT': f1(e4r('path', 'kNN term', 'natural')), 'phModPose': f1(e4r('path', 'Orientation term', 'modality')),
          'phNatPose': f1(e4r('path', 'Orientation term', 'natural')), 'phRotPose': f1(max(e4t('path', 'Orientation term', t) for t in ('rot90', 'rot180', 'vflip'))),
          'phWtlKnnT': '{}/{}/{}'.format(*E4['path']['boot']['kNN term']['wtl']), 'phModCfsKnn': f1(e4r('path', 'CFS-kNN', 'modality'))})
M['xrAcc'] = f1(np.mean([h['acc'] for h in E4['chest']['heads'].values()]))
C10G = ['c10n4', 'c10x4', 'c10r25', 'c10r50', 'c10r100']
M.update({'rgAccMin': f1(min(np.mean([h['acc'] for h in E4[t]['heads'].values()]) for t in C10G)), 'rgAccMax': f1(max(np.mean([h['acc'] for h in E4[t]['heads'].values()]) for t in C10G)),
          'rgPoseVfMin': f1(min(E4[t]['rows']['Orientation term']['flip'] for t in C10G)), 'rgPoseVfMax': f1(max(E4[t]['rows']['Orientation term']['flip'] for t in C10G)),
          'rgKnnTVfMin': f1(min(E4[t]['rows']['kNN term']['flip'] for t in C10G)), 'rgKnnTVfMax': f1(max(E4[t]['rows']['kNN term']['flip'] for t in C10G)),
          'rgNllVfMin': f1(min(E4[t]['rows']['NLL']['flip'] for t in C10G)), 'rgNllVfMax': f1(max(E4[t]['rows']['NLL']['flip'] for t in C10G)),
          'rgNllNatMin': f1(min(E4[t]['rows']['NLL']['natural'] for t in C10G)), 'rgNllNatMax': f1(max(E4[t]['rows']['NLL']['natural'] for t in C10G)),
          'rgNllUpVfMax': f1(max(E4[t]['rows']['NLL + log P(up|x)']['flip'] for t in C10G)), 'rgNllUpVfR': f1(E4['c10r100']['rows']['NLL + log P(up|x)']['flip']),
          'rgNllVfR': f1(E4['c10r100']['rows']['NLL']['flip']), 'rgPoseVfR': f1(E4['c10r100']['rows']['Orientation term']['flip']),
          'rgOursNatMin': f1(min(E4[t]['rows']['Ours']['natural'] for t in C10G)), 'rgOursNatMax': f1(max(E4[t]['rows']['Ours']['natural'] for t in C10G)),
          'rgKnnTNatMin': f1(min(E4[t]['rows']['kNN term']['natural'] for t in C10G)), 'rgKnnTNatMax': f1(max(E4[t]['rows']['kNN term']['natural'] for t in C10G))})
M['priorWinMinNat'] = str(min(wtl_nat_v(m)[0] for m in COMPLETE)); M['priorWinMaxNat'] = str(max(wtl_nat_v(m)[0] for m in COMPLETE))
M['priorLossMaxNat'] = str(max(wtl_nat_v(m)[2] for m in PRIOR))
M['priorWinMinNatName'] = LABEL[min(COMPLETE, key=lambda m: wtl_nat_v(m)[0])]
M['wtlTunedNat'] = wtl_nat('kNN tuned'); M['wtlKnnTNat'] = wtl_nat('Ours content'); M['wtlPoseNat'] = wtl_nat('Ours pose')
M['winKnnTNat'] = str(wtl_nat_v('Ours content')[0]); M['lossKnnTNat'] = str(wtl_nat_v('Ours content')[2])
M['gainKnnTNat'] = f1(np.mean([J['pairs'][t][o]['rows']['Ours content']['diff'] for t, o in NATP]))
M['knnTCten'] = f1(J['ids']['edmc10']['rows']['Ours content']['mean']); M['knnTIN'] = f1(J['ids']['in64']['rows']['Ours content']['mean'])
M['knnTMean'] = f1(ids_mean(J, 'Ours content'))
lostN = [(t, o) for t, o in NATP if any(J['pairs'][t][o]['rows'][r]['verdict'] == 'loss' for r in PRIOR if r in J['pairs'][t][o]['rows'])]   # lost to a prior detector (not to our own kNN rules)
lostOwn = [(t, o) for t, o in NATP if any(J['pairs'][t][o]['rows'][r]['verdict'] == 'loss' for r in ('kNN tuned', 'kNN (CT)'))]
# pairs on which the sum is below the better of its two terms by the bootstrap rule (loss against the kNN term or the orientation term)
belowB = [(t, o, min(J['pairs'][t][o]['rows'][r]['diff'] for r in ('Ours content', 'Ours pose'))) for t, o in NATP
          if any(J['pairs'][t][o]['rows'][r]['verdict'] == 'loss' for r in ('Ours content', 'Ours pose'))]
belowB.sort(key=lambda x: x[2])
SETN = {'cifar10_test': 'CIFAR-10', 'cifar100_test': 'CIFAR-100', 'celeba_test': 'CelebA', 'svhn_test': 'SVHN', 'in64_inaturalist': 'iNaturalist', 'textures': 'Textures',
        'tin_val': 'Tiny-ImageNet', 'lsun_fix': 'LSUN-FIX', 'imagenet_fix': 'ImageNet-FIX', 'imagenet32_val': 'ImageNet32', 'places365_val': 'Places365', 'in64_ssb_hard': 'SSB-hard',
        'in64_ninco': 'NINCO', 'in64_textures': 'Textures', 'in64_openimage_o': 'OpenImage-O', 'in64_places365': 'Places365', 'in64_celeba_test': 'CelebA'}
IDN = {'edmc10': 'CIFAR-10', 'l6svhn': 'SVHN', 'dpceleba': 'CelebA', 'edmc100': 'CIFAR-100', 'in64': 'ImageNet-64', 'fmnist': 'FashionMNIST'}
M['lostNatList'] = ', '.join(f'{IDN[t]}$\\to${SETN[o]}' for t, o in lostN)
M['nLostNat'] = str(len(lostN)); M['nNoLossNat'] = str(len(NATP) - len(lostN))
M['lostOwnList'] = ', '.join(f'{IDN[t]}$\\to${SETN[o]}' for t, o in lostOwn); M['nLostOwn'] = str(len(lostOwn))
M['nSumBelowBest'] = str(len(belowB)); M['sumBelowBestMaxGap'] = f1(round(max(pair(belowB[0][0], belowB[0][1], r) for r in ('Ours content', 'Ours pose')), 1) - round(pair(belowB[0][0], belowB[0][1]), 1)) if belowB else '0'
M['sumBelowBestList'] = '; '.join(f"{IDN[t]}$\\to${SETN[o]} {f1(pair(t, o))} vs {f1(max(pair(t, o, 'Ours content'), pair(t, o, 'Ours pose')))}\\%" for t, o, _ in belowB[:3])
partial = [r for r in NEWROWS if len(NEWTIERS[r]) < 6]                # rows whose runs on some ID datasets are still pending (footnote of Table 2)
def partial_note(r):                                                 # the ID datasets a partial row covers: 'so far' while runs are missing, its scope once they are complete
    done = NEWTIERS[r]; nat_pairs = NEWJ['rows'][r]['nat_pairs']
    if r == 'Kamkari' and all(t in done for t in KAM_ALL): return f"{LABEL[r]}: the five 32$\\times$32 datasets, {nat_pairs} pairs (no public configuration for ImageNet-64)"
    return f"{LABEL[r]}: " + ', '.join(IDN[t] for t in TAGS if t in done) + ' so far'
M['partialRowsNote'] = '' if not partial else '; ' + '; '.join(partial_note(r) for r in partial)
ROWS_ALL = ['kNN tuned', 'kNN (CT)'] + PRIOR
lost = [(t, o) for t, P in J['pairs'].items() for o, p in P.items() if any(p['rows'][r]['verdict'] == 'loss' for r in ROWS_ALL if r in p['rows'])]
M['nNoLoss'] = str(int(M['nPairs']) - len(lost)); M['nLost'] = str(len(lost))
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
fpp = min([r for r in PRIOR if r in FP['mean']], key=lambda r: FP['mean'][r]['fpr95']); M['FPRbestPriorName'] = LABEL[fpp]; M['FPRbestPrior'] = f1(FP['mean'][fpp]['fpr95'])
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


# ---- Table 1 (6 ID tiers, final split): columns = 6 ID means, Mean, Near (4 near-OOD pairs), Flip (probe, unmarked), W/T/L, NFE, CPU
def marks(vals):
    """per column: best -> light blue + bold, second -> underline (displayed values; ties share the mark; None = no entry)"""
    r = [None if v is None else round(v, 1) for v in vals]; u = sorted(set(x for x in r if x is not None), reverse=True)
    return [('' if x is None else ('b' if x == u[0] else ('s' if len(u) > 1 and x == u[1] else ''))) for x in r]
def cell(v, m):
    if v is None: return '--'
    return {'b': f'\\cellcolor{{firstblue}}\\textbf{{{f1(v)}}}', 's': f'\\underline{{{f1(v)}}}', '': f1(v)}[m]
GROUPS = [('\\emph{Likelihood, score, reconstruction and dimension statistics of the diffusion model}', EIGHT + NEWROWS),
          ('\\emph{Features of the diffusion model}', ['CFS', 'CFS-kNN', 'kNN (CT)', 'kNN tuned']),
          ('\\emph{Networks trained on the ID images}', ['RotCls4'] + (['CSI'] if CSI else []))]
def rowvals(r):
    if r == 'CSI':
        ids = [CSI['ids'][t]['csi_nat'] if t in CSI['ids'] else None for t in TAGS]
        return ids + [float(np.mean([v for v in ids if v is not None])), None, float(np.mean([v['csi'] for v in CSI['upside_down'].values()]))]
    ids = [J['ids'][t]['rows'][r]['mean'] if r in J['ids'][t]['rows'] else None for t in TAGS]
    return ids + [ids_mean(J, r), near_mean(r), float(np.mean([vf_of(J, t, r) for t in TAGS if r in J['pairs'][t][VF[t]]['rows']]))]
rows = [r for _, rs in GROUPS for r in rs]
cols = {r: rowvals(r) for r in rows}
cols['ours'] = [J['ids'][t]['ours'] for t in TAGS] + [J['mean_ours'], near_mean(), float(np.mean(vo))]
order = rows + ['ours']; mk = {r: [] for r in order}
for j in range(9):
    mm = marks([cols[r][j] for r in order]) if j != 8 else [''] * len(order)      # the Flip column (a probe) is not marked
    for r, m in zip(order, mm): mk[r].append(m)
def wtl_cell(r):
    """wins/ties/losses by the bootstrap rule, followed by the wins and losses of at least one AUROC point in parentheses"""
    if r == 'CSI':
        w = CSI['wtl_nat']; nat = [p for t in CSI['pairs'] for p in CSI['pairs'][t].values() if p['kind'] == 'natural']
        bw = sum(p['verdict'] == 'win' and p['diff'] >= 1 for p in nat); bl = sum(p['verdict'] == 'loss' and p['diff'] <= -1 for p in nat)
        return f"{w['wins']}/{w['ties']}/{w['losses']} ({bw}/{bl})$^{{\\S}}$"
    v = wtl_nat_v(r); bw, bl = big_wtl(r); return f'{v[0]}/{v[1]}/{v[2]} ({bw}/{bl})' + ('' if sum(v) == len(NATP) else '$^{\\S}$')
NCOL = 13
L = ['% written by gen/mk_ijcai.py']
for gi, (title, rs) in enumerate(GROUPS):
    L.append(('\\midrule' if gi else '') + f'\\multicolumn{{{NCOL}}}{{@{{}}l}}{{{title}}}\\\\')
    for r in rs:
        lab = LABEL[r] + ('$^{\\S}$' if r == 'CSI' or (r in NEWTIERS and len(NEWTIERS[r]) < 6) else '')
        L.append(lab + ' & ' + ' & '.join(cell(v, m) for v, m in zip(cols[r], mk[r])) + f" & {wtl_cell(r)} & {PASSES[r]} & {fsec(SECROW[r])}\\\\")
L.append('\\midrule')
L.append('\\textbf{Upright kNN (ours)} & ' + ' & '.join(cell(v, m) for v, m in zip(cols['ours'], mk['ours'])) + f" & -- & 1 & {fsec(SECROW['ours'])}\\\\")
for key, lab in (('Ours content', '\\quad kNN term only'), ('Ours pose', '\\quad orientation term only')):
    vals = [J['ids'][t]['rows'][key]['mean'] for t in TAGS] + [ids_mean(J, key), near_mean(key), float(np.mean([vf_of(J, t, key) for t in TAGS]))]
    L.append(f'{lab} & ' + ' & '.join(f1(v) for v in vals) + f" & {wtl_cell(key)} & 1 & --\\\\")
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
R64 = ['Ours', 'kNN term', 'Orientation term', 'kNN basic', 'CFS', 'CFS kNN', 'RotCls4', 'NLL', 'TT', 'MSMA*', 'DiffPath']   # the weaker rows (IC, GEPC, EigenScore) are in the supplement
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
