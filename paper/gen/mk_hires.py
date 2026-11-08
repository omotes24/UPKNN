# mk_hires.py -- Table 4 of the paper (resolution-controlled ImageNet: ADM 256x256 in pixels, EDM2-XS 512x512 and DiT-XL/2 256x256 in
#   latents; PROTOCOL_HIRES.md, PROTOCOL_HIRES_DIT.md), its macros (numbers_hires.tex) and the per-set table of the supplement
#   (supp_hires.tex), from gen/data/hires_ana.json (analysis/final/hires/ana/hires_ana.py).  While the runs are going, placeholders.
import os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.dirname(HERE); D = os.path.join(HERE, 'data')
SRC = os.environ.get('HIRES_JSON', f'{D}/hires_ana.json')
TIERS = [('in256c', 'ADM, 256$\\times$256, pixels'), ('e512', 'EDM2-XS, 512$\\times$512, latents'), ('dit256', 'DiT-XL/2, 256$\\times$256, latents')]
ROWS = ['Ours', 'kNN term', 'Orientation term', 'kNN basic', 'CFS', 'CFS kNN', 'MSMA*', 'DiffPath', 'GEPC']
LAB = {'Ours': 'Upright kNN', 'kNN term': '\\quad kNN term', 'Orientation term': '\\quad orient.\\ term', 'kNN basic': 'kNN (basic)',
       'CFS': 'CFS (diag.)', 'CFS kNN': 'CFS (kNN)', 'MSMA*': 'MSMA$^{*}$', 'DiffPath': 'DiffPath', 'GEPC': 'GEPC'}
ABL = ('kNN term', 'Orientation term')
NEAR = ['ssb_hard', 'ninco']; FAR = ['inaturalist', 'textures', 'openimage_o']; VF = 'imagenet_test__vflip'; CS = ['imagenet_v2', 'imagenet_c', 'imagenet_r']
SN = {'ssb_hard': 'SSB-hard', 'ninco': 'NINCO', 'inaturalist': 'iNat', 'textures': 'Tex.', 'openimage_o': 'OI-O', VF: 'Flip',
      'imagenet_v2': 'V2', 'imagenet_c': 'C', 'imagenet_r': 'R'}
f1 = lambda x: f'{x:.1f}'


def marks(vals):
    r = [round(v, 1) for v in vals]; u = sorted(set(r), reverse=True)
    return [('b' if x == u[0] else ('s' if len(u) > 1 and x == u[1] else '')) for x in r]


def cell(v, m): return {'b': f'\\cellcolor{{firstblue}}\\textbf{{{f1(v)}}}', 's': f'\\underline{{{f1(v)}}}', '': f1(v)}[m]


M = {}
if not os.path.exists(SRC):
    open(f'{OUT}/tab_hires.tex', 'w').write('% written by gen/mk_hires.py (placeholder)\n\\multicolumn{10}{@{}l}{\\pending{runs on the GPU server; this table is filled from hires\\_ana.json}}\\\\\n')
    open(f'{OUT}/supp_hires.tex', 'w').write('% written by gen/mk_hires.py (placeholder)\n\\multicolumn{11}{@{}l}{\\pending{runs on the GPU server}}\\\\\n')
    M['hiresStatus'] = 'pending'
else:
    J = json.load(open(SRC)); have = [t for t, _ in TIERS if t in J]
    get = lambda t, r: J[t]['common'].get(r) if t in J else None
    cols = []                                                                  # (tier, key) for Near, Far, Flip
    for t, _ in TIERS: cols += [(t, 'near'), (t, 'far'), (t, VF)]
    val = lambda t, r, k: (None if get(t, r) is None else (get(t, r)[k]['auroc'] if k == VF else get(t, r)[k]))
    mk = {r: [] for r in ROWS}
    for t, k in cols:
        rr = [r for r in ROWS if r not in ABL and val(t, r, k) is not None]; mm = dict(zip(rr, marks([val(t, r, k) for r in rr])))
        for r in ROWS: mk[r].append(mm.get(r, ''))
    L = ['% written by gen/mk_hires.py']
    for r in ROWS:
        if all(val(t, r, 'near') is None for t, _ in TIERS): continue
        cells = [('--' if val(t, r, k) is None else (f1(val(t, r, k)) if r in ABL else cell(val(t, r, k), m))) for (t, k), m in zip(cols, mk[r])]
        L.append(LAB[r] + ' & ' + ' & '.join(cells) + '\\\\')
        if r == 'Orientation term': L.append('\\midrule')
    open(f'{OUT}/tab_hires.tex', 'w').write('\n'.join(L) + '\n')
    # appendix: every set, every row, per tier, with W/T/L of Upright kNN over the five OOD sets
    L = ['% written by gen/mk_hires.py']
    for t, lab in TIERS:
        if t not in J: continue
        L.append(f'\\midrule\\multicolumn{{11}}{{@{{}}l}}{{\\emph{{{lab}}}: {min(J[t]["n_common"].values())}--{max(J[t]["n_common"].values())} common images per set}}\\\\')
        for r in ROWS:
            g = get(t, r)
            if g is None: continue
            w = '--' if r == 'Ours' else '{}/{}/{}'.format(*J[t]['boot'][r]['wtl'])
            L.append(LAB[r] + ' & ' + ' & '.join(f1(g[s]['auroc']) for s in NEAR + FAR + [VF] + CS) + f' & {w}\\\\')
    open(f'{OUT}/supp_hires.tex', 'w').write('\n'.join(L) + '\n')
    M['hiresStatus'] = 'done'
    for t, short in (('in256c', 'Adm'), ('e512', 'Edm'), ('dit256', 'Dit')):
        if t not in J: continue
        o = J[t]['common']['Ours']; M[f'h{short}Near'] = f1(o['near']); M[f'h{short}Far'] = f1(o['far']); M[f'h{short}Vf'] = f1(o[VF]['auroc'])
        for k, nm in (('kNN term', 'KnnT'), ('Orientation term', 'Pose'), ('kNN basic', 'KnnB')):
            if k in J[t]['common']:
                g = J[t]['common'][k]; M[f'h{short}Near{nm}'] = f1(g['near']); M[f'h{short}Far{nm}'] = f1(g['far']); M[f'h{short}Vf{nm}'] = f1(g[VF]['auroc'])
        prior = [r for r in ('CFS', 'CFS kNN', 'MSMA*', 'DiffPath', 'GEPC') if r in J[t]['common']]
        if prior:
            bn = max(prior, key=lambda r: J[t]['common'][r]['near']); bf = max(prior, key=lambda r: J[t]['common'][r]['far'])
            M[f'h{short}NearPrior'] = f1(J[t]['common'][bn]['near']); M[f'h{short}NearPriorName'] = LAB[bn]
            M[f'h{short}FarPrior'] = f1(J[t]['common'][bf]['far']); M[f'h{short}FarPriorName'] = LAB[bf]
            M[f'h{short}WtlMin'] = str(min(J[t]['boot'][r]['wtl'][0] for r in prior)); M[f'h{short}LossMax'] = str(max(J[t]['boot'][r]['wtl'][2] for r in prior))
            M[f'h{short}NearVerdicts'] = ', '.join(f"{LAB[r]}: {J[t]['boot'][r]['near']['verdict']}" for r in prior)
        M[f'h{short}Ncom'] = str(min(J[t]['n_common'].values()))
        M[f'h{short}Ssb'] = f1(o['ssb_hard']['auroc']); M[f'h{short}Ninco'] = f1(o['ninco']['auroc'])
DG = f'{D}/hires_diag.json'                                                   # post-hoc diagnostics (code/hires_diag.py)
if os.path.exists(DG):
    Q = json.load(open(DG)); F = Q['feat']
    sh = [F['dit256'][f'L{l}_top_channel_share_median'] for l in range(3)]
    M['hDitTopShareMin'] = f'{100 * min(sh):.0f}'; M['hDitTopShareMax'] = f'{100 * max(sh):.0f}'
    M['hDitCosMed'] = f"{F['dit256']['pair_cos_quantiles_1_50_99'][1]:.3f}"
    oc = [F[t]['pair_cos_quantiles_1_50_99'][1] for t in ('in256c', 'e512')]; M['hUnetCosMin'] = f'{min(oc):.2f}'; M['hUnetCosMax'] = f'{max(oc):.2f}'
    P = {k: v for k, v in Q['pix'].items() if k in NEAR + FAR}
    M['hPixSsb'] = f1(max(P['ssb_hard']['hf_auroc'], P['ssb_hard']['blk_auroc'])); M['hPixNinco'] = f1(max(P['ninco']['hf_auroc'], P['ninco']['blk_auroc']))
    M['hPixMax'] = f1(max(max(v['hf_auroc'], v['blk_auroc']) for v in P.values()))
E7 = f'{D}/e7_ana.json'                                                       # PROTOCOL_HIRES2.md: identical resampling history (analysis/final/e7/ana/e7_ana.py)
if os.path.exists(E7) and 'in256d' in json.load(open(E7)):
    Z = json.load(open(E7)); X = Z['cross']; R2 = [r for r in ('Ours', 'kNN term', 'Orientation term', 'kNN basic', 'CFS', 'CFS kNN') if r in X['rows']]
    L = ['% written by gen/mk_hires.py (PROTOCOL_HIRES2.md)']
    for r in R2:
        a, b = X['rows'][r]['in256c'], X['rows'][r]['in256d']; bt = X['boot'][r]
        def pc(s):
            d = bt[s]['diff']; v = bt[s]['verdict']; mark = {'win': '$^{+}$', 'loss': '$^{-}$', 'tie': ''}[v]
            return f"{f1(a[s]['auroc'] if s not in ('near', 'far') else a[s])} & {f1(b[s]['auroc'] if s not in ('near', 'far') else b[s])}{mark}"
        L.append(LAB[r] + ' & ' + ' & '.join(pc(s) for s in NEAR + ['near'] + FAR + ['far', VF]) + '\\\\')
        if r == 'Orientation term': L.append('\\midrule')
    open(f'{OUT}/supp_hires2.tex', 'w').write('\n'.join(L) + '\n')
    o = X['rows']['Ours']; k = X['rows']['kNN term']; kb = X['rows']['kNN basic']
    M.update({'hTwoNcom': str(min(X['n_common'].values())),
              'hTwoSsbC': f1(o['in256c']['ssb_hard']['auroc']), 'hTwoSsbD': f1(o['in256d']['ssb_hard']['auroc']), 'hTwoSsbVerdict': X['boot']['Ours']['ssb_hard']['verdict'],
              'hTwoNincoC': f1(o['in256c']['ninco']['auroc']), 'hTwoNincoD': f1(o['in256d']['ninco']['auroc']), 'hTwoNincoVerdict': X['boot']['Ours']['ninco']['verdict'],
              'hTwoNearC': f1(o['in256c']['near']), 'hTwoNearD': f1(o['in256d']['near']), 'hTwoNearVerdict': X['boot']['Ours']['near']['verdict'],
              'hTwoFarC': f1(o['in256c']['far']), 'hTwoFarD': f1(o['in256d']['far']), 'hTwoFarVerdict': X['boot']['Ours']['far']['verdict'],
              'hTwoVfC': f1(o['in256c'][VF]['auroc']), 'hTwoVfD': f1(o['in256d'][VF]['auroc']),
              'hTwoInatC': f1(o['in256c']['inaturalist']['auroc']), 'hTwoInatD': f1(o['in256d']['inaturalist']['auroc']), 'hTwoInatVerdict': X['boot']['Ours']['inaturalist']['verdict'],
              'hTwoOioC': f1(o['in256c']['openimage_o']['auroc']), 'hTwoOioD': f1(o['in256d']['openimage_o']['auroc']), 'hTwoOioVerdict': X['boot']['Ours']['openimage_o']['verdict'],
              'hTwoTexC': f1(o['in256c']['textures']['auroc']), 'hTwoTexD': f1(o['in256d']['textures']['auroc']),
              'hTwoSsbKnnTC': f1(k['in256c']['ssb_hard']['auroc']), 'hTwoSsbKnnTD': f1(k['in256d']['ssb_hard']['auroc']),
              'hTwoOioKnnBC': f1(kb['in256c']['openimage_o']['auroc']), 'hTwoOioKnnBD': f1(kb['in256d']['openimage_o']['auroc']),
              'hTwoNearDiff': f"{X['boot']['Ours']['near']['diff']:+.1f}", 'hTwoSsbDiff': f"{X['boot']['Ours']['ssb_hard']['diff']:+.1f}"})
    if 'CFS' in X['rows']:
        c = X['rows']['CFS']; M['hTwoNearCfsD'] = f1(c['in256d']['near']); M['hTwoSsbCfsD'] = f1(c['in256d']['ssb_hard']['auroc'])
        M['hTwoNearCfsC'] = f1(c['in256c']['near'])
    # the in256d tier on its own (all rows, common images), as Table S-hires-by-set
    T = Z['in256d']; M['hTwoStatus'] = 'done'
    M['hTwoWtlCfs'] = '{}/{}/{}'.format(*T['boot']['CFS']['wtl']) if 'CFS' in T['boot'] else '--'
    M['hTwoWtlCfsKnn'] = '{}/{}/{}'.format(*T['boot']['CFS kNN']['wtl']) if 'CFS kNN' in T['boot'] else '--'
else:
    M['hTwoStatus'] = 'pending'
assert all(not any(ch.isdigit() for ch in k) for k in M), [k for k in M if any(ch.isdigit() for ch in k)]
with open(f'{OUT}/numbers_hires.tex', 'w') as f:
    f.write('% written by gen/mk_hires.py\n')
    for k, v in M.items(): f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
print(M)
