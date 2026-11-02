# mk_q_hires.py -- the job queue of PROTOCOL_HIRES.md (runs/jobs/Q_hires.txt, consumed by eu/experiments/runq2.py on every GPU), in priority
#   order: data, latents, Upright kNN seeds, CFS seeds, then DiffPath / MSMA* / GEPC on the controlled 256 images (stored-run settings).
import os
ROOT = os.path.expanduser('~/upknn'); Q = f'{ROOT}/runs/jobs/Q_hires.txt'
assert not os.path.exists(Q), 'exists'
J = ['python code/prep_c224.py > logs/hires/prep.log 2>&1 && python code/mk_sets_hires.py >> logs/hires/prep.log 2>&1 && touch logs/hires/PREP_DONE',
     'until [ -f logs/hires/PREP_DONE ]; do sleep 60; done; python code/enc_e512.py --gpu {gpu} > logs/hires/enc.log 2>&1']
J += [f'bash code/hires_seed.sh in256c {{gpu}} {s} > logs/hires/seed_in256c_{s}.log 2>&1' for s in (0, 1, 2)]
J += [f'bash code/hires_seed.sh e512 {{gpu}} {s} > logs/hires/seed_e512_{s}.log 2>&1' for s in (0, 1, 2)]
J += [f'bash code/hires_cfs.sh {t} {{gpu}} {s} > logs/hires/cfs_{t}_{s}.log 2>&1' for t in ('e512', 'in256c') for s in (0, 1, 2)]
W = 'until [ -f logs/hires/PREP_DONE ]; do sleep 60; done; DG_FP16=0 python code/run_adm.py '
sets = [('hc_imagenet_test', '--n 1000', 'hc_imagenet_test__cal', ''), ('hc_imagenet_test', '--n 3000 --complement --cap 500', 'hc_imagenet_test', ''),
        ('hc_imagenet_test', '--n 3000 --complement --cap 500', 'hc_imagenet_test__vflip', ' --pre vflip')]
sets += [(f'hc_{s}', '--n 0 --complement --cap 500', f'hc_{s}', '') for s in ('ssb_hard', 'ninco', 'inaturalist', 'textures', 'openimage_o', 'imagenet_v2', 'imagenet_c', 'imagenet_r')]
for meth, bs in (('diffpath', 4), ('msma', 8), ('gepc', 2)):
    for d, sel, name, pre in sets:
        if meth == 'msma':
            J.append(W + f'code/extract.py --gpu {{gpu}} --model adm:256 --K 20 --smin 0.01 --smax 10 --draws 2 --hutch 0 --tfs id --nobands --bs {bs} --data {d} {sel}{pre} --out runs/base/F_in256c/{name}.npz > logs/hires/msma_{name}.log 2>&1')
        else:
            J.append(W + f'code/baselines.py --gpu {{gpu}} --method {meth} --model adm:256 --bs {bs} --data {d} {sel}{pre} --out runs/base2/{meth}_in256c/{name}.npz > logs/hires/{meth}_{name}.log 2>&1')
os.makedirs(f'{ROOT}/runs/jobs', exist_ok=True); os.makedirs(f'{ROOT}/logs/hires', exist_ok=True)
for d in ('runs/base/F_in256c', 'runs/base2/diffpath_in256c', 'runs/base2/gepc_in256c'): os.makedirs(f'{ROOT}/{d}', exist_ok=True)
open(Q, 'w').write('\n'.join(J) + '\n'); print(len(J), 'jobs ->', Q)
