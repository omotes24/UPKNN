# mk_inputs_fresh.py -- inputs tree of the held-out ("fresh") evaluation for analysis/baselines_eval.py: the per-set statistics of the
#   comparison methods recomputed on the held-out images (final/data/runs/{base,base2,nll}/*_fresh*) and the calibration files
#   (<id>__cal, ID development images) of the original inputs, linked into final/inputs_fresh/{base,base2,nll}.
#   usage: python mk_inputs_fresh.py   (then SPLIT=fresh python permethod.py)
import os, sys
from ana_common import *
SRC = os.environ['EUPHONIUM_INPUTS']; OUT = f'{SCR}/final/inputs_fresh'
RAW = ['diffpath', 'gepc', 'eigen']


def link(src, dst):
    if os.path.lexists(dst): os.remove(dst)
    os.symlink(os.path.realpath(src), dst)


missing = []
for tag, T in TIERS.items():
    # MSMA score norms and PF-ODE likelihoods: one directory per tier, the cal file from the original run
    for kind, run_src, run_dst in (('base', f'{DATA}/runs/base/F_{tag}_fresh', f'{OUT}/base/F_{tag}'), ('nll', f'{DATA}/runs/nll/F_{tag}_fresh', f'{OUT}/nll/F_{tag}')):
        os.makedirs(run_dst, exist_ok=True)
        cal = f'{SRC}/{kind}/{RUN["pair"][tag] if kind == "x" else "F_" + tag}/{T["id"]}__cal.npz'
        if not os.path.exists(cal): cal = f'{SRC}/{kind}/{NLLRUN.get(tag, "F_" + tag)}/{T["id"]}__cal.npz'
        if os.path.exists(cal): link(cal, f'{run_dst}/{T["id"]}__cal.npz')
        else: missing.append(cal)
        if os.path.isdir(run_src):
            for f in os.listdir(run_src):
                if f.endswith('.npz'): link(f'{run_src}/{f}', f'{run_dst}/{f}')
        else: missing.append(run_src)
    # rot90 likelihoods (exact ratio)
    rs = f'{DATA}/runs/nll/F_{tag}_fresh_rot90'
    if os.path.isdir(rs):
        rd = f'{OUT}/nll/F_{tag}_rot90'; os.makedirs(rd, exist_ok=True)
        for f in os.listdir(rs):
            if f.endswith('.npz'): link(f'{rs}/{f}', f'{rd}/{f}')
    # DiffPath / GEPC / EigenScore (seed 0 and seeds 1, 2 as their own directories)
    for raw in RAW:
        for seed in (0, 1, 2):
            src = f'{DATA}/runs/base2/{raw}_{tag}_fresh' + (f'_s{seed}' if seed else '')
            if not os.path.isdir(src):
                if seed == 0 or raw != 'diffpath': missing.append(src)
                continue
            dst = f'{OUT}/base2/{raw}_{tag}' + (f'_s{seed}' if seed else ''); os.makedirs(dst, exist_ok=True)
            cal = f'{SRC}/base2/{raw}_{tag}/{T["id"]}__cal.npz'
            if seed: cal_s = f'{DATA}/runs/base2/{raw}_{tag}_s{seed}/{T["id"]}__cal.npz'; cal = cal_s if os.path.exists(cal_s) else cal
            link(cal, f'{dst}/{T["id"]}__cal.npz')
            for f in os.listdir(src):
                if f.endswith('.npz'): link(f'{src}/{f}', f'{dst}/{f}')
for d in ('complexity', 'feat'):
    if not os.path.lexists(f'{OUT}/{d}'): os.symlink(os.path.realpath(f'{SRC}/{d}'), f'{OUT}/{d}')
print('inputs_fresh built at', OUT); print('missing:', *missing, sep='\n  ')
