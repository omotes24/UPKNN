#!/bin/bash
# leak-free rotation heads (k0; g and gq) with temperature, for every ID; waits for the rotated bank of kx_extract_rot.py
cd ~/upknn
PY=${PY:-python}  
run() { g=$1; shift; for t in "$@"; do
  while [ ! -f runs/kx/rotb/s0/$t/k0/bank__idx.npy ]; do sleep 20; done
  for f in gq g; do $PY code/kx_rothead2.py --tag $t --gpu $g --k 0 --feat $f --bankdir runs/kx/rotb --tagout R >> logs/kx/headsR_$t.log 2>&1; $PY code/kx_rh2scores2.py --tag $t --gpu $g --k 0 --feat $f --tagout R >> logs/kx/headsR_$t.log 2>&1; done; done; }
run 2 dpceleba in64 l6svhn &
run 3 edmc10 edmc100 fmnist &
wait; echo ALL_DONE >> logs/kx/headsR.done
