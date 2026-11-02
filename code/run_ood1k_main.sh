#!/bin/bash
# run_ood1k_main.sh -- the frozen ImageNet-64 detectors on the OpenOOD v1.5 ImageNet-1K benchmark (sets ood1k- of runs/ok/sets_in64.json):
#   Upright kNN (ok_extract -> kx_applyhead -> kx_export, unchanged), Vertical Limit (vl_fextract at sigma_1 and the deepest level ->
#   vl_fscore_x), the basic kNN, per noise seed on GPUs 0-2; CFS (lambda = 7, 3 seeds) and the rotation classifier on GPU 3.
cd ~/upknn
PY=${PY:-python}  
NAMES=$($PY -c "import json; print(','.join(r[0] for r in json.load(open('runs/ok/sets_in64.json')) if r[0].startswith('ood1k-')))")
up() { g=$1; s=$2
  $PY code/ok_extract.py --tag in64 --gpu $g --sets runs/ok/sets_in64.json --ks 0 --seed $s --only $NAMES >> logs/ood1k/upright_s$s.log 2>&1
  $PY code/kx_applyhead.py --tag in64 --gpu $g --seed $s --k 0 --feat gq --tagout R --prefix ood1k- >> logs/ood1k/upright_s$s.log 2>&1
  $PY code/kx_export.py in64 $g $s ood1k- >> logs/ood1k/upright_s$s.log 2>&1
  $PY code/vl_fextract.py --tag in64 --gpu $g --seed $s --k 0 --level 3 --prefix ood1k- --nobank 1 >> logs/ood1k/vl_s$s.log 2>&1
  $PY code/vl_fscore_x.py in64 $g $s ood1k >> logs/ood1k/vl_s$s.log 2>&1
  $PY code/ood1k_knn.py --seed $s --gpu $g >> logs/ood1k/knn_s$s.log 2>&1
  echo UP_DONE_$s >> logs/ood1k/main.done; }
cfs() { g=$1
  $PY code/mk_rotcls_idx.py >> logs/ood1k/rotcls.log 2>&1
  ROTCLS_COM=runs/ood1k/rotcls_idx.json ROTCLS_SCORES=scores_ood1k $PY eu/experiments/rotcls_eval.py in64 $g >> logs/ood1k/rotcls.log 2>&1
  for s in 0 1 2; do
    DG_FP16=0 $PY code/cfs_extract.py --tag in64 --gpu $g --seed $s --lam 7 --prefix ood1k- --out runs/cfs_l7 >> logs/ood1k/cfs.log 2>&1
    $PY code/cfs_score.py --tag in64 --gpu $g --seed $s --run runs/cfs_l7 >> logs/ood1k/cfs.log 2>&1; done
  echo CFS_DONE >> logs/ood1k/main.done; }
up 0 0 & up 1 1 & up 2 2 & cfs 3 &
wait
echo ALL_MAIN_DONE >> logs/ood1k/main.done
