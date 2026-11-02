#!/bin/bash
# rotated bank (noise not rotated), sigma_1 only, noise seeds 1 and 2; then the leak-free heads (k0; g, gq) for those seeds
cd ~/upknn
PY=${PY:-python}  
run() { g=$1; shift; for t in "$@"; do for s in 1 2; do
  $PY code/kx_extract_rot.py --tag $t --gpu $g --ks 0 --seed $s >> logs/kx/extract_rot_s${s}_$t.log 2>&1
  for f in gq g; do $PY code/kx_rothead2.py --tag $t --gpu $g --k 0 --feat $f --bankdir runs/kx/rotb --tagout R --seed $s >> logs/kx/headsR_s${s}_$t.log 2>&1; done
  done; done; }
run 0 edmc10 &
run 1 edmc100 dpceleba &
run 2 in64 l6svhn &
run 3 fmnist &
wait; echo ALL_DONE >> logs/kx/extract_rot_s12.done
