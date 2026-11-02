#!/bin/bash
# e4_tier.sh <tag> <gpu> -- one tier of PROTOCOL_E4.md on one GPU: finishes an interrupted training run if needed, then Upright kNN
#   (noise seeds 0-2) and kNN (basic), then the comparison rows (chest, path: CFS seeds 0-2, NLL, MSMA*, DiffPath + their '__cal' runs;
#   c10*: NLL).  Finished steps are skipped (done.txt markers, existing outputs).
cd ~/upknn; PY=${PY:-python}; tag=$1; g=$2; L=logs/e4; mkdir -p $L
case $tag in
  chest)   M=models/chest_m;   MS=own:models/chest_m/final.pt;    ID=g32_chest_test; TR=g32_chest_train;;
  path)    M=models/path_m;    MS=own:models/path_m/final.pt;     ID=path_test;      TR=path_train;;
  c10n4)   M=;                 MS=own:models/c10_m/snap_4M.pt;    ID=cifar10_test;   TR=;;
  c10x4)   M=models/c10_xflip; MS=own:models/c10_xflip/final.pt;  ID=cifar10_test;   TR=;;
  c10r25)  M=;                 MS=own:models/c10_rot25/final.pt;  ID=cifar10_test;   TR=;;
  c10r50)  M=;                 MS=own:models/c10_rot50/final.pt;  ID=cifar10_test;   TR=;;
  c10r100) M=;                 MS=own:models/c10_rot100/final.pt; ID=cifar10_test;   TR=;;
  *) echo "unknown tag $tag"; exit 1;;
esac
if [ -n "$M" ] && [ ! -f runs/$M/final.pt ]; then $PY code/e4_resume.py $M $g >> $L/train_$tag.log 2>&1 || exit 1; fi
[ -f runs/ok/sets_$tag.json ] || $PY code/mk_sets_e4.py >> $L/sets.log 2>&1 || exit 1
echo "tier $tag model $MS" >> $L/tier_$tag.log
for s in 0 1 2; do
  grep -qx "E4_DONE_${tag}_$s" $L/done.txt 2>/dev/null && continue
  $PY code/run_e4.py code/kx_extract_rot.py --tag $tag --gpu $g --ks 0 --seed $s >> $L/rot_${tag}_s$s.log 2>&1 || exit 1
  $PY code/run_e4.py code/ok_extract.py --tag $tag --gpu $g --sets runs/ok/sets_$tag.json --ks 0 --seed $s >> $L/ok_${tag}_s$s.log 2>&1 || exit 1
  $PY code/kx_rothead2_lm.py --tag $tag --gpu $g --k 0 --feat gq --bankdir runs/kx/rotb --tagout R --seed $s --prefix e4- >> $L/head_${tag}_s$s.log 2>&1 || exit 1
  $PY code/kx_export_x.py $tag $g $s e4- >> $L/export_${tag}_s$s.log 2>&1 || exit 1
  $PY code/e4_knn.py --tag $tag --seed $s --gpu $g >> $L/knn_${tag}_s$s.log 2>&1 || exit 1
  echo E4_DONE_${tag}_$s >> $L/done.txt
done
if [ -n "$TR" ]; then
  for s in 0 1 2; do
    grep -qx "E4_CFS_${tag}_$s" $L/done.txt 2>/dev/null && continue
    DG_FP16=0 $PY code/run_e4.py code/cfs_extract.py --tag $tag --gpu $g --seed $s --lam 7 --out runs/cfs_l7 --prefix e4- >> $L/cfs_${tag}_s$s.log 2>&1 || exit 1
    $PY code/run_e4.py code/cfs_score.py --tag $tag --gpu $g --seed $s --run runs/cfs_l7 >> $L/cfs_${tag}_s$s.log 2>&1 || exit 1
    echo E4_CFS_${tag}_$s >> $L/done.txt
  done
fi
J=eu/runs/jobs/e4_idx
SETS=$($PY -c "import json; print(' '.join(r[0][3:] + ':' + r[1] + ':' + r[3] for r in json.load(open('runs/ok/sets_$tag.json'))))")
[ -n "$TR" ] && SETS="$SETS ${ID}__cal:$TR:none"
for e in $SETS; do
  IFS=: read n d pre <<< "$e"
  if [ "$n" = "${ID}__cal" ]; then ix=$J/${tag}__${TR}__cal.json; else ix=$J/${tag}__$n.json; fi
  if [ "$n" != "${ID}__cal" ] && [ ! -f runs/nll/E4_$tag/$n.npz ]; then
    SR_ROOT=$HOME/upknn $PY eu/experiments/nll.py --gpu $g --model $MS --data $d --idx $ix --pre $pre --out runs/nll/E4_$tag/$n.npz --grid log --steps 64 --smin 0.002 --bs 100 >> $L/nll_$tag.log 2>&1 || exit 1
  fi
  if [ -n "$TR" ]; then
    [ -f runs/base/E4_$tag/$n.npz ] || DG_FP16=0 $PY code/extract.py --gpu $g --model $MS --data $d --idx $ix --pre $pre --out runs/base/E4_$tag/$n.npz --K 20 --smin 0.004 --smax 10 --draws 2 --hutch 4 --tfs id --nobands --bs 250 >> $L/msma_$tag.log 2>&1 || exit 1
    [ -f runs/base2/diffpath_E4_$tag/$n.npz ] || SR_ROOT=$HOME/upknn SR_FP16=0 $PY eu/experiments/baselines.py --gpu $g --method diffpath --model $MS --data $d --idx $ix --pre $pre --bs 250 --out runs/base2/diffpath_E4_$tag/$n.npz >> $L/diffpath_$tag.log 2>&1 || exit 1
  fi
done
echo E4_ALL_DONE_$tag >> $L/done.txt
