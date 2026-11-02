#!/bin/bash
# run_ablate2b.sh -- reruns kx_ablate2.py v2 (orbit-bank d in single precision) on the development split and selection data (seed 0)
cd ~/upknn
PY=${PY:-python}  
run() { g=$1; shift; for t in "$@"; do $PY code/kx_ablate2.py $t $g >> logs/kx/ablate2b.log 2>&1; done; }
run 0 edmc10 l6svhn &
run 1 edmc100 dpceleba &
run 2 in64 &
run 3 fmnist &
wait
echo ABL2B_DONE >> logs/kx/ablate2b.log
