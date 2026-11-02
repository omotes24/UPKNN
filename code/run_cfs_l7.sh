#!/bin/bash
# run_cfs_l7.sh -- CFS(1x2) at lambda = 7 (the canonical level of the official repository configs), noise seed 0, for the choice of lambda on
#   the development split (fresh); output runs/cfs_l7
cd ~/upknn
PY=${PY:-python}  
run() { g=$1; shift; for t in "$@"; do
  DG_FP16=0 $PY code/cfs_extract.py --tag $t --gpu $g --seed 0 --lam 7 --out runs/cfs_l7 >> logs/cfs/l7_$t.log 2>&1; done; }
run 0 edmc10 fmnist &
run 1 l6svhn dpceleba &
run 2 edmc100 in64 &
wait
echo CFS_L7_DONE >> logs/cfs/run_l7.log
