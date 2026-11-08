#!/bin/bash
# run_vl_final.sh -- Vertical Limit on one split for the given IDs, noise seeds 0-2, from runs/vl/PROTOCOL_VL.json:
#   features of the vertical level (vl_fextract.py: bank + the split's sets, at sigma_0 if a0 = 1 and at sigma* if w* > 0), then the scores
#   (vl_fscore.py -> runs/vl/export/<tag>_s<seed>_<split>.npz).   final2 needs the md5 of the frozen protocol as the 4th argument.
#   usage: bash code/run_vl_final.sh <gpu> <split: fresh|final2> "<tags>" [md5]
cd ~/upknn
PY=${PY:-python}  
gpu=$1; split=$2; tags=$3; md5=$4
for t in $tags; do
  read lv ks <<< $($PY -c "import json; P=json.load(open('runs/vl/PROTOCOL_VL.json'))['$t']; ks=([0] if P['a0'] else []) + ([P['kstar']] if P['kstar'] is not None and P['wstar'] > 0 else []); print(P['level'], ','.join(map(str, ks)) or 'none')")
  for s in 0 1 2; do
    if [ "$ks" != "none" ]; then
      for k in ${ks//,/ }; do
        if [ "$split" = "final2" ]; then
          $PY code/vl_fextract.py --tag $t --gpu $gpu --seed $s --k $k --level $lv --prefix final2- --frozen $md5 >> logs/vl/fextract_${t}.log 2>&1
        else
          $PY code/vl_fextract.py --tag $t --gpu $gpu --seed $s --k $k --level $lv --prefix fresh- >> logs/vl/fextract_${t}.log 2>&1
        fi
      done
    fi
    $PY code/vl_fscore.py $t $gpu $s $split >> logs/vl/fscore_${t}.log 2>&1
  done
done
echo "RUN_VL_FINAL_DONE $split $tags" >> logs/vl/final.done
