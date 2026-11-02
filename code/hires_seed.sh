#!/bin/bash
# hires_seed.sh <tag: in256c | e512> <gpu> <seed> -- Upright kNN for one noise seed of a tier of PROTOCOL_HIRES.md
cd ~/upknn; PY=${PY:-python}; tag=$1; g=$2; s=$3; L=logs/hires; mkdir -p $L
until [ -f $L/PREP_DONE ]; do sleep 60; done
[ $tag = e512 ] && until [ -f $L/ENC512_DONE ]; do sleep 60; done
[ $tag = dit256 ] && until [ -f $L/ENCDIT_DONE ]; do sleep 60; done
case $tag in in256c) PFX=q256-;; e512) PFX=q512-;; dit256) PFX=qdit-;; esac
D=runs/ok/s$s/$tag/k0; B=runs/kx/rotb/s$s/$tag/k0
SETS=$($PY -c "import json; print(','.join(r[0] for r in json.load(open('runs/ok/sets_$tag.json'))))")
if [ $tag = in256c ]; then
  $PY code/run_hires.py code/kx_extract_rot.py --tag $tag --gpu $g --ks 0 --seed $s || exit 1
  OK_VIEWS=1 $PY code/run_hires.py code/ok_extract_v0.py --tag $tag --gpu $g --sets runs/ok/sets_$tag.json --ks 0 --seed $s --only $SETS || exit 1
  for f in bank__L0.npy bank__L1.npy bank__L2.npy bank__idx.npy; do [ -e $D/$f ] || ln -s $(realpath $B/$f) $D/$f; done
else
  OK_VIEWS=1 $PY code/run_hires.py code/ok_extract_v0.py --tag $tag --gpu $g --sets runs/ok/sets_$tag.json --ks 0 --seed $s --only $SETS || exit 1
  $PY code/assemble_rotb.py $tag $s || exit 1
fi
$PY code/kx_rothead2.py --tag $tag --gpu $g --k 0 --feat gq --bankdir runs/kx/rotb --tagout R --seed $s --prefix $PFX || exit 1
$PY code/kx_export_x.py $tag $g $s $PFX || exit 1
$PY code/knn_basic.py --tag $tag --seed $s --gpu $g --prefix $PFX || exit 1
echo HIRES_DONE_${tag}_$s >> $L/done.txt
