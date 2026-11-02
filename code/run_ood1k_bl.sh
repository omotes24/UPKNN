#!/bin/bash
# run_ood1k_bl.sh <gpu> <marker> -- waits until run_ood1k_main.sh has finished its work on this GPU (<marker> in logs/ood1k/main.done),
#   then works through the shared queue eu/jobs/Q_ood1k_bl.txt (mk_ood1k_bl.py) with runq2.py.
cd ~/upknn; export PATH=$PATH
until grep -qx "$2" logs/ood1k/main.done 2>/dev/null; do sleep 60; done
python eu/experiments/runq2.py eu/jobs/Q_ood1k_bl.txt $1 >> logs/ood1k/blq_gpu$1.log 2>&1
echo BLQ_DONE_$1 >> logs/ood1k/main.done
