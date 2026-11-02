#!/bin/bash
# run_ts2.sh -- two_subspace2.py over tiers, seeds and splits (CPU); variants ('all') for seed 0 on final2
cd ~/upknn; export PATH=$PATH; mkdir -p logs/two
for t in edmc10 l6svhn dpceleba edmc100 in64 fmnist; do
  python code/two_subspace2.py $t 0 final2 cpu all > logs/two/ts2_${t}_s0_final2.log 2>&1
done
for sd in 1 2; do for t in edmc10 l6svhn dpceleba edmc100 in64 fmnist; do
  python code/two_subspace2.py $t $sd final2 cpu > logs/two/ts2_${t}_s${sd}_final2.log 2>&1
done; done
for sd in 0 1 2; do for t in edmc10 l6svhn dpceleba edmc100 in64 fmnist; do
  python code/two_subspace2.py $t $sd fresh cpu > logs/two/ts2_${t}_s${sd}_fresh.log 2>&1
done; done
echo ALL_DONE > logs/two/TS2_DONE
