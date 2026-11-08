#!/bin/bash
# run_vl_hooks3.sh -- the mirror tests per block at sigma_1 in single precision for the six IDs (vl_hooks3.py), three GPUs
cd ~/upknn
PY=${PY:-python}  
( for t in edmc10 l6svhn; do $PY code/vl_hooks3.py --tag $t --gpu 0 > logs/vl/hooks3_$t.log 2>&1; done; echo H0 >> logs/vl/hooks3.done ) &
( for t in edmc100 fmnist; do $PY code/vl_hooks3.py --tag $t --gpu 1 > logs/vl/hooks3_$t.log 2>&1; done; echo H1 >> logs/vl/hooks3.done ) &
( for t in in64 dpceleba; do $PY code/vl_hooks3.py --tag $t --gpu 2 > logs/vl/hooks3_$t.log 2>&1; done; echo H2 >> logs/vl/hooks3.done ) &
wait
echo ALL_HOOKS3_DONE >> logs/vl/hooks3.done
