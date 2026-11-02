#!/bin/bash
# run_in256.sh <gpu> <seed> -- Upright kNN on the ImageNet-256 tier for one noise seed (PROTOCOL_IN256.md):
#   rotated bank views -> test-set features (h256-*) -> bank view 0 linked from the rotated-view run -> head -> scores, then CFS (seed>0)
cd ~/upknn; PY=${PY:-python}; g=$1; s=$2; L=logs/in256; mkdir -p $L
SETS=h256-imagenet_test,h256-imagenet_test__vflip,h256-ssb_hard,h256-ninco,h256-inaturalist,h256-textures,h256-openimage_o
$PY code/run_in256.py code/kx_extract_rot.py --tag in256 --gpu $g --ks 0 --seed $s >> $L/rot_s$s.log 2>&1 || exit 1
OK_VIEWS=1 $PY code/run_in256.py code/ok_extract_v0.py --tag in256 --gpu $g --sets runs/ok/sets_in256.json --ks 0 --seed $s --only $SETS >> $L/ok_s$s.log 2>&1 || exit 1
D=runs/ok/s$s/in256/k0; B=runs/kx/rotb/s$s/in256/k0
for f in bank__L0.npy bank__L1.npy bank__L2.npy bank__idx.npy; do [ -e $D/$f ] || ln -s $(realpath $B/$f) $D/$f; done
$PY code/kx_rothead2.py --tag in256 --gpu $g --k 0 --feat gq --bankdir runs/kx/rotb --tagout R --seed $s --prefix h256- >> $L/head_s$s.log 2>&1 || exit 1
$PY code/kx_export_x.py in256 $g $s h256- >> $L/export_s$s.log 2>&1 || exit 1
echo IN256_DONE_$s >> $L/done.txt
until grep -qx CS_DONE $L/done.txt 2>/dev/null; do sleep 60; done
$PY code/kx_applyhead.py --tag in256 --gpu $g --seed $s --k 0 --feat gq --tagout R --prefix c256- >> $L/head_s$s.log 2>&1
$PY code/kx_export_x.py in256 $g $s c256- >> $L/export_s$s.log 2>&1
$PY code/in256_knn.py --seed $s --gpu $g >> $L/knn_s$s.log 2>&1
echo IN256_POST_$s >> $L/done.txt
if [ $s -gt 0 ]; then
  DG_FP16=0 $PY code/run_in256.py code/cfs_extract.py --tag in256 --gpu $g --seed $s --lam 7 --out runs/cfs_l7 --prefix h256-,c256- >> $L/cfs_s$s.log 2>&1
  $PY code/cfs_score.py --tag in256 --gpu $g --seed $s --run runs/cfs_l7 >> $L/cfs_s$s.log 2>&1
  echo CFS_DONE_$s >> $L/done.txt
fi
