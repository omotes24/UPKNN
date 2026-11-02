#!/bin/bash
# run_in256_g3.sh <gpu> -- the covariate-shift sets of the ImageNet-256 tier (prep + features for seeds 0-2), then CFS seed 0
cd ~/upknn; PY=${PY:-python}; g=$1; L=logs/in256; mkdir -p $L
$PY code/prep_hr_cs.py >> $L/prep_cs.log 2>&1 || exit 1
for s in 0 1 2; do
  OK_VIEWS=1 $PY code/run_in256.py code/ok_extract_v0.py --tag in256 --gpu $g --sets runs/ok/sets_in256.json --ks 0 --seed $s --only c256-imagenet_v2,c256-imagenet_c,c256-imagenet_r >> $L/okcs_s$s.log 2>&1 || exit 1
done
echo CS_DONE >> $L/done.txt
DG_FP16=0 $PY code/run_in256.py code/cfs_extract.py --tag in256 --gpu $g --seed 0 --lam 7 --out runs/cfs_l7 --prefix h256-,c256- >> $L/cfs_s0.log 2>&1
$PY code/cfs_score.py --tag in256 --gpu $g --seed 0 --run runs/cfs_l7 >> $L/cfs_s0.log 2>&1
echo CFS_DONE_0 >> $L/done.txt
