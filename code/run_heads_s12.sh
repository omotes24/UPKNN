#!/bin/bash
cd ~/upknn
PY=${PY:-python}  
run() { g=$1; shift; for t in "$@"; do for s in 1 2; do
  while [ ! -f runs/kx/rotb/s$s/$t/k0/bank__idx.npy ]; do sleep 20; done
  $PY code/kx_rothead2.py --tag $t --gpu $g --k 0 --feat gq --bankdir runs/kx/rotb --tagout R --seed $s --prefix devall-,fresh- >> logs/kx/headsR2_s${s}_$t.log 2>&1
  $PY code/kx_export.py $t $g $s fresh- >> logs/kx/export.log 2>&1; done; done; }
run 0 edmc10 l6svhn &
run 2 edmc100 in64 &
run 3 fmnist dpceleba &
wait; echo ALL_DONE >> logs/kx/heads_s12.done
