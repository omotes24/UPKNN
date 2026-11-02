# ana_common.py -- shared loaders for the final-resume analyses (cloud side)
import os, sys, json
import numpy as np
SCR = os.environ.get('UPKNN_ANALYSIS', os.path.expanduser('~/upknn/analysis'))
REPO = f'{SCR}/eu_build/Euphonium'
os.environ.setdefault('EUPHONIUM_INPUTS', f'{SCR}/eu_build/inputs_full')
sys.path.insert(0, f'{REPO}/analysis'); sys.path.insert(0, REPO)
from common import TIERS, NAT, RUN, BASELINES, ID_NAME, feat, first_pair, atom_score, sub, auroc, fpr95, common_idx, set_name, POOL, DEVRUN
DATA = f'{SCR}/final/data'
COM = common_idx()
ATOMS = [('rot90', 0), ('rot90', 6), ('rot180', 0)]


def setname(tag, s):
    return TIERS[tag]['id'] if s == '__id__' else s


def eu_run(tag, name, seed=0):
    """per-image outputs of score_euphonium.py (--pred_diff) on the common images: dict with eu, log_ratios, e_up, e_rot, pd, a_up, idx"""
    z = np.load(f'{DATA}/runs/eu/s{seed}/{tag}/{name}.npz')
    return {k: z[k] for k in z.files}


def on_common(tag, s, score, idx):
    return sub(np.asarray(score), np.asarray(idx), COM[tag][s if s in COM[tag] else '__id__'])
