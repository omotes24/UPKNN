#!/bin/bash
# ct_worker.sh GPU -- runs the (tag, k) jobs of runs/ct/jobs.txt one by one, each claimed through an atomic mkdir lock
GPU=$1
: && cd ~/upknn || exit 1
mkdir -p runs/ct/locks logs/ct
while read -r tag k; do
  [ -z "$tag" ] && continue
  if mkdir "runs/ct/locks/${tag}_k${k}" 2>/dev/null; then
    echo "$(date '+%F %T') GPU$GPU start $tag k$k"
    if [ "$tag" = edmc100 ]; then
      DG_FP16=0 python code/ct_extract.py --tag "$tag" --gpu "$GPU" --ks "$k" < /dev/null >> "logs/ct/${tag}_k${k}.log" 2>&1
    else
      python code/ct_extract.py --tag "$tag" --gpu "$GPU" --ks "$k" < /dev/null >> "logs/ct/${tag}_k${k}.log" 2>&1
    fi
    echo "$(date '+%F %T') GPU$GPU end $tag k$k rc=$?"
  fi
done < runs/ct/jobs.txt
echo "$(date '+%F %T') GPU$GPU WORKER_DONE"
