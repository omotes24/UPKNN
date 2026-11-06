#!/bin/bash
# hires2_fix.sh <gpu> <seed> -- finishes one seed of PROTOCOL_HIRES2.md after kx_export_x.py had failed with KeyError 'in256d'
#   (its per-tier table CH lacked the tier; the entry ('L012', 1000), the frozen ImageNet-64 configuration as for in256c, was added).
#   Waits for the head of that seed (hires2_seed.sh), then runs the two remaining steps of hires2_seed.sh: export and kNN (basic).
cd ~/upknn; PY=${PY:-python}; tag=in256d; g=$1; s=$2; L=logs/hires2; PFX=q256d-
until grep -q "KX_ROTHEAD_DONE" $L/seed_$s.log; do sleep 60; done
$PY code/kx_export_x.py $tag $g $s $PFX >> $L/fix_$s.log 2>&1 || exit 1
$PY code/knn_basic.py --tag $tag --seed $s --gpu $g --prefix $PFX >> $L/fix_$s.log 2>&1 || exit 1
echo HIRES2_DONE_${tag}_$s >> $L/done.txt
echo HIRES2_FIX_DONE $s
