#!/bin/bash
# hires2_cfs.sh <gpu> <seed> -- CFS (lambda 7, fp32) for the in256d tier of PROTOCOL_HIRES2.md (as hires_cfs.sh for in256c)
cd ~/upknn; PY=${PY:-python}; tag=in256d; g=$1; s=$2; L=logs/hires2; mkdir -p $L
until [ -f $L/PREP_DONE ]; do sleep 60; done
PFX=q256d-
DG_FP16=0 $PY code/run_hires2.py code/cfs_extract.py --tag $tag --gpu $g --seed $s --lam 7 --out runs/cfs_l7 --prefix $PFX || exit 1
$PY code/cfs_score.py --tag $tag --gpu $g --seed $s --run runs/cfs_l7 || exit 1
echo CFS2_DONE_${tag}_$s >> $L/done.txt
