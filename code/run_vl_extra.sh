#!/bin/bash
# run_vl_extra.sh -- additional development-split experiments for the final resume (never final2):
#   vl_hooks2.py (mirror tests V/H per block at sigma_0, 6 IDs), deep-level features at sigma_0 for the IDs whose VL has a0 = 0
#   (l6svhn, fmnist; fresh split, seeds 0-2), and vl_ctrl.py (the two controls) for the 6 IDs.
cd ~/upknn
PY=${PY:-python}  
mkdir -p logs/vl
( for t in edmc10 l6svhn fmnist; do $PY code/vl_hooks2.py --tag $t --gpu 0 > logs/vl/hooks2_$t.log 2>&1; done; echo G0 >> logs/vl/extra.done ) &
( for t in edmc100 dpceleba; do $PY code/vl_hooks2.py --tag $t --gpu 1 > logs/vl/hooks2_$t.log 2>&1; done; echo G1 >> logs/vl/extra.done ) &
( for t in l6svhn fmnist; do for s in 0 1 2; do $PY code/vl_fextract.py --tag $t --gpu 2 --seed $s --k 0 --level 2 --prefix fresh- >> logs/vl/fextract_k0_$t.log 2>&1; done; done
  echo FEXTRACT_K0_DONE >> logs/vl/extra.done
  $PY code/vl_hooks2.py --tag in64 --gpu 2 > logs/vl/hooks2_in64.log 2>&1; echo G2 >> logs/vl/extra.done ) &
( for t in edmc10 edmc100 dpceleba in64; do $PY code/vl_ctrl.py --tag $t --gpu 3 > logs/vl/ctrl_$t.log 2>&1; done
  while ! grep -q FEXTRACT_K0_DONE logs/vl/extra.done 2>/dev/null; do sleep 20; done
  for t in l6svhn fmnist; do $PY code/vl_ctrl.py --tag $t --gpu 3 > logs/vl/ctrl_$t.log 2>&1; done; echo G3 >> logs/vl/extra.done ) &
wait
echo ALL_EXTRA_DONE >> logs/vl/extra.done
