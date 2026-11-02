# run_e5.py -- runs a script of the pipeline (kx_extract_rot.py, cfs_extract.py, cfs_score.py) with the ImageNet-64 bank tiers of
#   PROTOCOL_E5.md registered in ct_extract_defs.SPEC (same EDM2-XS model as 'in64', other bank arrays); the script itself is unchanged.
import os, sys, runpy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ct_extract_defs
for k in (10, 25, 50):
    ct_extract_defs.SPEC[f'in64h{k}'] = ('edm2:edm2-img64-xs-uncond-2147483-0.100.pkl', f'in64_hfbank_{k}k', 125)
script = sys.argv[1]; sys.argv = [os.path.basename(script)] + sys.argv[2:]
runpy.run_path(os.path.join(HERE, os.path.basename(script)), run_name='__main__')
