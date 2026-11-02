# run_in256.py -- runs a script of the Upright kNN pipeline (ok_extract.py, kx_extract_rot.py, cfs_extract.py, ...) with the ADM
#   ImageNet-256 unconditional model as the tag 'in256' (PROTOCOL_IN256.md): adds SPEC['in256'] = ('adm:256', 'hr_imagenet_bank', 8)
#   to ct_extract_defs and the spec 'adm:256' to dgcore.load_model (admwrap.load_adm; fp16 unless DG_FP16=0), then runs the script
#   unchanged.   usage: python code/run_in256.py code/<script>.py <its arguments, with --tag in256>
import os, sys, runpy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import dgcore, ct_extract_defs, admwrap
ct_extract_defs.SPEC['in256'] = ('adm:256', 'hr_imagenet_bank', 8)
_load_model = dgcore.load_model


def load_model(spec, device):
    if spec == 'adm:256': return admwrap.load_adm(device, fp16=os.environ.get('DG_FP16', '1') != '0')
    return _load_model(spec, device)


dgcore.load_model = load_model
script = sys.argv[1]
if os.path.basename(script) == 'cfs_extract.py':                # the 32-image fp32 probe batch of CFS does not fit an 11-GB GPU at 256x256
    script = 'cfs_extract_x.py'; os.environ.setdefault('CFS_PROBE_CHUNK', '4')
sys.argv = [os.path.basename(script)] + sys.argv[2:]
runpy.run_path(os.path.join(HERE, os.path.basename(script)), run_name='__main__')
