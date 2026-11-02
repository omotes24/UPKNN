#!/bin/bash
cd ~/upknn
PY=${PY:-python}  
run() { g=$1; shift; for t in "$@"; do $PY code/kx_extract_rot.py --tag $t --gpu $g --ks 0,6 >> logs/kx/extract_rot_$t.log 2>&1; done; }
run 0 edmc10 fmnist &
run 1 edmc100 &
run 2 in64 dpceleba &
run 3 l6svhn &
wait; echo ALL_DONE >> logs/kx/extract_rot.done
