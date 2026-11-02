# mk_ood1k_bl.py -- the comparison methods of Table 1 on the OpenOOD v1.5 ImageNet-1K benchmark at 64x64 (sets ood1k- of
#   runs/ok/sets_in64.json, prep_ood1k.py).  The likelihood / statistics baselines are expensive, so they run on a fixed random subsample
#   of 1,000 images of every benchmark set (RandomState(20261102), drawn from the set's own index list); Upright kNN, Vertical Limit, the
#   kNN rows, CFS and the rotation classifier run on the full sets (run_ood1k_main.sh) and are compared with the baselines on the same
#   1,000 images.  Calibration fits of the baselines are those of the main evaluation (nothing is refitted).
#   Methods and settings exactly as in the final evaluation (mk_final2_jobs.py, ImageNet-64 rows): PF-ODE likelihood (NLL / IC / TT),
#   MSMA-style score norms, DiffPath, GEPC (seed 0), EigenScore (seed 0); plus the PNG code lengths of the three new data sets (IC).
#   writes runs/ood1k/s1k/<data>.json, runs/ood1k/s1k/s1k_idx.json ({"in64": {"__id__": idx, "<data>": idx}}) and prints the job lines.
#   usage (from ~/upknn): python code/mk_ood1k_bl.py > eu/jobs/Q_ood1k_bl.txt
import os, sys, json
import numpy as np
ROOT = os.path.expanduser('~/upknn'); N1 = 1000
S = [r for r in json.load(open(f'{ROOT}/runs/ok/sets_in64.json')) if r[0].startswith('ood1k-')]
O = f'{ROOT}/runs/ood1k/s1k'; os.makedirs(O, exist_ok=True)
com = {}
for name, data, idx, pre in S:
    assert pre == 'none' and len(idx) >= N1, name
    sub = np.sort(np.random.RandomState(20261102).choice(np.asarray(idx, np.int64), N1, replace=False)).tolist()
    json.dump(sub, open(f'{O}/{data}.json', 'w')); com['__id__' if data == 'in64_imagenet_test' else data] = sub
json.dump({'in64': com}, open(f'{O}/s1k_idx.json', 'w'))
print('# s1k index files:', {k: len(v) for k, v in com.items()}, file=sys.stderr)
MODEL = 'edm2:edm2-img64-xs-uncond-2147483-0.100.pkl'; E = 'SR_ROOT=$HOME/upknn '; J = 'runs/ood1k/s1k'
DATA = [r[1] for r in S]
L = []
def job(cmd, log): L.append(f'{cmd} > logs/ood1k/bl_{log}.log 2>&1')
job(f'{E}python eu/experiments/complexity.py in64_imagenet_v2 in64_imagenet_c in64_imagenet_r', 'complexity')
for d in DATA:
    job(f'{E}python eu/experiments/nll.py --gpu {{gpu}} --model {MODEL} --data {d} --idx {J}/{d}.json --out runs/nll/F_in64_ood1k/{d}.npz '
        f'--grid log --steps 64 --smin 0.002 --bs 16', f'nll_{d}')
for d in DATA:
    job(f'DG_FP16=0 python code/extract.py --gpu {{gpu}} --model {MODEL} --data {d} --idx {J}/{d}.json --out runs/base/F_in64_ood1k/{d}.npz '
        f'--K 20 --smin 0.004 --smax 10 --draws 2 --hutch 4 --tfs id --nobands --bs 64', f'msma_{d}')
for d in DATA:
    job(f'{E}SR_FP16=0 python eu/experiments/baselines.py --gpu {{gpu}} --method diffpath --model {MODEL} --data {d} --idx {J}/{d}.json '
        f'--bs 64 --out runs/base2/diffpath_in64_ood1k/{d}.npz', f'diffpath_{d}')
for m, bs, extra in (('gepc', 16, ''), ('eigen', 16, ' --reps 2 --iters 3 --neig 3')):
    for d in DATA:
        job(f'{E}SR_FP16=0 python eu/experiments/baselines.py --gpu {{gpu}} --method {m} --model {MODEL} --data {d} --idx {J}/{d}.json '
            f'--bs {bs}{extra} --seed 0 --out runs/base2/{m}_in64_ood1k/{d}.npz', f'{m}_{d}')
assert all('\n' not in j for j in L)
print('\n'.join(L)); print('#', len(L), 'jobs', file=sys.stderr)
