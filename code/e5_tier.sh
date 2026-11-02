#!/bin/bash
# e5_tier.sh <tag: in64h10 | in64h25 | in64h50> <gpu> -- PROTOCOL_E5.md on one GPU: rotated bank views, a head trained on this bank,
#   Upright kNN / kNN (basic) on the ood1k- sets (test features of the frozen run reused via links), then CFS (lambda 7) with this bank.
cd ~/upknn; PY=${PY:-python}; tag=$1; g=$2; L=logs/e5; mkdir -p $L
[ -f runs/ok/sets_$tag.json ] && [ -f data/npy/in64_hfbank_50k.npy ] || $PY code/mk_e5.py >> $L/mk.log 2>&1 || exit 1
for s in 0 1 2; do
  grep -qx "E5_DONE_${tag}_$s" $L/done.txt 2>/dev/null && continue
  $PY code/run_e5.py code/kx_extract_rot.py --tag $tag --gpu $g --ks 0 --seed $s >> $L/rot_${tag}_s$s.log 2>&1 || exit 1
  D=runs/ok/s$s/$tag/k0; B=runs/kx/rotb/s$s/$tag/k0; S0=runs/ok/s$s/in64/k0; mkdir -p $D
  for f in bank__L0.npy bank__L1.npy bank__L2.npy bank__idx.npy; do [ -e $D/$f ] || ln -s $(realpath $B/$f) $D/$f; done
  for f in $S0/ood1k-*__L[012].npy $S0/ood1k-*__idx.npy; do [ -e $D/$(basename $f) ] || ln -s $(realpath $f) $D/$(basename $f); done
  $PY code/kx_rothead2_lm.py --tag $tag --gpu $g --k 0 --feat gq --bankdir runs/kx/rotb --tagout R --seed $s --prefix ood1k- >> $L/head_${tag}_s$s.log 2>&1 || exit 1
  $PY code/e5_export.py $tag $g $s >> $L/export_${tag}_s$s.log 2>&1 || exit 1
  echo E5_DONE_${tag}_$s >> $L/done.txt
done
for s in 0 1 2; do
  grep -qx "E5_CFS_${tag}_$s" $L/done.txt 2>/dev/null && continue
  DG_FP16=0 $PY code/run_e5.py code/cfs_extract.py --tag $tag --gpu $g --seed $s --lam 7 --prefix ood1k- --out runs/cfs_l7 >> $L/cfs_${tag}_s$s.log 2>&1 || exit 1
  $PY code/run_e5.py code/cfs_score.py --tag $tag --gpu $g --seed $s --run runs/cfs_l7 >> $L/cfs_${tag}_s$s.log 2>&1 || exit 1
  echo E5_CFS_${tag}_$s >> $L/done.txt
done
echo E5_ALL_DONE_$tag >> $L/done.txt
