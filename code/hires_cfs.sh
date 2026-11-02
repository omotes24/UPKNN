#!/bin/bash
# hires_cfs.sh <tag> <gpu> <seed> -- CFS (lambda 7, fp32) for a tier of PROTOCOL_HIRES.md
cd ~/upknn; PY=${PY:-python}; tag=$1; g=$2; s=$3; L=logs/hires; mkdir -p $L
until [ -f $L/PREP_DONE ]; do sleep 60; done
[ $tag = e512 ] && until [ -f $L/ENC512_DONE ]; do sleep 60; done
if [ $tag = in256c ]; then PFX=q256-; else PFX=q512-; fi
DG_FP16=0 $PY code/run_hires.py code/cfs_extract.py --tag $tag --gpu $g --seed $s --lam 7 --out runs/cfs_l7 --prefix $PFX || exit 1
$PY code/cfs_score.py --tag $tag --gpu $g --seed $s --run runs/cfs_l7 || exit 1
echo CFS_DONE_${tag}_$s >> $L/done.txt
