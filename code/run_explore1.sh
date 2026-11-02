#!/bin/bash
# run_explore1.sh -- round 1 of kx_explore.py on the four GPUs (tmux session dg_kx), selection + fresh, seed 0
cd ~/upknn
PY=${PY:-python}  
SP=$(cat code/kx_spaces1.txt)
run() { g=$1; shift; for t in "$@"; do $PY code/kx_explore.py --tag $t --gpu $g --spaces "$SP" --out runs/kx/explore1_$t.json >> logs/kx/explore1_$t.log 2>&1; done; }
run 0 edmc10 &
run 1 edmc100 &
run 2 in64 fmnist &
run 3 l6svhn dpceleba &
wait; echo ALL_DONE >> logs/kx/explore1.done
