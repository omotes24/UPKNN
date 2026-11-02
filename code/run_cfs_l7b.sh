#!/bin/bash
# run_cfs_l7b.sh -- CFS(1x2) at lambda = 7 (chosen over lambda = 5 on the development split, the setting more favourable to CFS):
#   noise seeds 1, 2 (seed 0 by run_cfs_l7.sh) and the heads (cfs_score.py) for all three seeds; output runs/cfs_l7
cd ~/upknn
PY=${PY:-python}  
run() { g=$1; shift; for t in "$@"; do
  for s in 1 2; do DG_FP16=0 $PY code/cfs_extract.py --tag $t --gpu $g --seed $s --lam 7 --out runs/cfs_l7 >> logs/cfs/l7_$t.log 2>&1; done
  for s in 0 1 2; do $PY code/cfs_score.py --tag $t --gpu $g --seed $s --run runs/cfs_l7 >> logs/cfs/l7_$t.log 2>&1; done; done; }
run 0 edmc10 fmnist &
run 1 l6svhn dpceleba &
run 2 edmc100 &
run 3 in64 &
wait
echo CFS_L7B_DONE >> logs/cfs/run_l7.log
