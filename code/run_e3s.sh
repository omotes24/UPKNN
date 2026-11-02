#!/bin/bash
# run_e3s.sh <gpu> <seed> -- one noise seed of run_e3.sh (PROTOCOL_E3.md); finished steps are skipped by the scripts themselves
cd ~/upknn; PY=${PY:-python}; g=$1; s=$2; L=logs/e3; mkdir -p $L
$PY code/run_e3.py code/kx_extract_rot.py --tag eurosat --gpu $g --ks 0 --seed $s >> $L/rot_s$s.log 2>&1 || exit 1
$PY code/run_e3.py code/ok_extract.py --tag eurosat --gpu $g --sets runs/ok/sets_eurosat.json --ks 0 --seed $s >> $L/ok_s$s.log 2>&1 || exit 1
$PY code/kx_rothead2.py --tag eurosat --gpu $g --k 0 --feat gq --bankdir runs/kx/rotb --tagout R --seed $s --prefix e3- >> $L/head_s$s.log 2>&1 || exit 1
$PY code/kx_export_x.py eurosat $g $s e3- >> $L/export_s$s.log 2>&1 || exit 1
$PY code/e3_knn.py --seed $s --gpu $g >> $L/knn_s$s.log 2>&1 || exit 1
echo E3_DONE_$s >> $L/done.txt
