#!/bin/bash
# hires2_seed.sh <gpu> <seed> -- Upright kNN for one noise seed of the in256d tier of PROTOCOL_HIRES2.md (as hires_seed.sh for in256c)
cd ~/upknn; PY=${PY:-python}; tag=in256d; g=$1; s=$2; L=logs/hires2; mkdir -p $L
until [ -f $L/PREP_DONE ]; do sleep 60; done
PFX=q256d-
D=runs/ok/s$s/$tag/k0; B=runs/kx/rotb/s$s/$tag/k0
SETS=$($PY -c "import json; print(','.join(r[0] for r in json.load(open('runs/ok/sets_$tag.json'))))")
$PY code/run_hires2.py code/kx_extract_rot.py --tag $tag --gpu $g --ks 0 --seed $s || exit 1
OK_VIEWS=1 $PY code/run_hires2.py code/ok_extract_v0.py --tag $tag --gpu $g --sets runs/ok/sets_$tag.json --ks 0 --seed $s --only $SETS || exit 1
mkdir -p $D
for f in bank__L0.npy bank__L1.npy bank__L2.npy bank__idx.npy; do [ -e $D/$f ] || ln -s $(realpath $B/$f) $D/$f; done
$PY code/kx_rothead2.py --tag $tag --gpu $g --k 0 --feat gq --bankdir runs/kx/rotb --tagout R --seed $s --prefix $PFX || exit 1
$PY code/kx_export_x.py $tag $g $s $PFX || exit 1
$PY code/knn_basic.py --tag $tag --seed $s --gpu $g --prefix $PFX || exit 1
echo HIRES2_DONE_${tag}_$s >> $L/done.txt
