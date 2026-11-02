# run_e3.py -- runs a script of the Upright kNN pipeline with the EuroSAT U-Net registered as the tag 'eurosat' (PROTOCOL_E3.md):
#   adds SPEC['eurosat'] = ('own:models/eurosat_m/final.pt', 'eurosat_train', 250) to ct_extract_defs, then runs the script unchanged.
#   usage: python code/run_e3.py code/<script>.py <its arguments, with --tag eurosat>
import os, sys, runpy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ct_extract_defs
ct_extract_defs.SPEC['eurosat'] = ('own:models/eurosat_m/final.pt', 'eurosat_train', 250)
script = sys.argv[1]; sys.argv = [os.path.basename(script)] + sys.argv[2:]
runpy.run_path(os.path.join(HERE, os.path.basename(script)), run_name='__main__')
