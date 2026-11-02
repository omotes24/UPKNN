#!/bin/bash
# run_final2_uknn.sh -- THE single final2 evaluation of the frozen orientation-completed kNN (PROTOCOL_UKNN.md): applies the trained heads
#   of every seed to the final2 sets (no training) and writes the per-image scores; then packs runs/kx/export/*final2*.npz.
cd ~/upknn
PY=${PY:-python}  
run() { g=$1; shift; for t in "$@"; do for s in 0 1 2; do
  $PY code/kx_applyhead.py --tag $t --gpu $g --seed $s --k 0 --feat gq --tagout R --prefix final2- >> logs/kx/final2_uknn.log 2>&1
  $PY code/kx_export.py $t $g $s final2- >> logs/kx/final2_uknn.log 2>&1; done; done; }
run 0 edmc10 l6svhn &
run 1 edmc100 &
run 2 in64 dpceleba &
run 3 fmnist &
wait
tar czf /tmp/dg_kx_final2_$(date +%m%d_%H%M).tgz runs/kx/export/*final2*.npz && echo FINAL2_PACKED >> logs/kx/final2_uknn.log
