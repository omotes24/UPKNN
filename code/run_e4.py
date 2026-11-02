# run_e4.py -- runs a script of the Upright kNN pipeline (kx_extract_rot.py, ok_extract.py, cfs_extract.py) with the own U-Nets of
#   PROTOCOL_E4.md registered as tags in ct_extract_defs.SPEC; the script itself is unchanged.
#   usage: python code/run_e4.py code/<script>.py <its arguments, with --tag chest | path | c10n4 | c10x4 | c10r25 | c10r50 | c10r100>
import os, sys, runpy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ct_extract_defs
E4 = {'chest': ('own:models/chest_m/final.pt', 'g32_chest_train', 250), 'path': ('own:models/path_m/final.pt', 'path_train', 250),
      'c10n4': ('own:models/c10_m/snap_4M.pt', 'cifar10_train', 250), 'c10x4': ('own:models/c10_xflip/final.pt', 'cifar10_train', 250),
      'c10r25': ('own:models/c10_rot25/final.pt', 'cifar10_train', 250), 'c10r50': ('own:models/c10_rot50/final.pt', 'cifar10_train', 250),
      'c10r100': ('own:models/c10_rot100/final.pt', 'cifar10_train', 250)}
ct_extract_defs.SPEC.update(E4)
script = sys.argv[1]; sys.argv = [os.path.basename(script)] + sys.argv[2:]
runpy.run_path(os.path.join(HERE, os.path.basename(script)), run_name='__main__')
