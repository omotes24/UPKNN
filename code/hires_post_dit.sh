#!/bin/bash
# hires_post_dit.sh <gpu> <seed> -- the steps of hires_seed.sh after the feature extraction for dit256, with the head of
#   code/kx_rothead2_lm2.py (held-out logits in chunks: the 34,560-dim DiT-XL/2 features did not fit an 11-GB GPU); otherwise identical.
cd ~/upknn; PY=${PY:-python}; tag=dit256; g=$1; s=$2; L=logs/hires; PFX=qdit-
grep -q "ASSEMBLE_ROTB_DONE dit256 $s" $L/seed_dit256_$s.log || { echo "extraction of seed $s not finished"; exit 1; }
$PY code/kx_rothead2_lm2.py --tag $tag --gpu $g --k 0 --feat gq --bankdir runs/kx/rotb --tagout R --seed $s --prefix $PFX || exit 1
$PY code/kx_export_x.py $tag $g $s $PFX || exit 1
$PY code/knn_basic.py --tag $tag --seed $s --gpu $g --prefix $PFX || exit 1
echo HIRES_DONE_${tag}_$s >> $L/done.txt
