#!/bin/bash
# waits for the feature queue, then scores dev (D_) and final (F_) feature sets against the four-orientation banks
: && cd ~/upknn
while tmux ls 2>/dev/null | grep -q "^dg_F"; do sleep 30; done
for t in edmc10 l6svhn dpceleba edmc100 in64 fmnist; do
  python code/fscores2.py runs/fd/D_$t/bank4.npz runs/fd/D_$t runs/fs2/D_$t
  python code/fscores2.py runs/fd/D_$t/bank4.npz runs/fd/F_$t runs/fs2/F_$t
done
echo FS2_ALL_DONE
