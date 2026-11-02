# run_hires.py -- runs a script of the Upright kNN pipeline (ok_extract_v0.py, kx_extract_rot.py, cfs_extract.py) for the tiers of
#   PROTOCOL_HIRES.md: 'in256c' (ADM 256, controlled images hc_*) and 'e512' (EDM2-XS-512 uncond on the latents lz512_*).
#   * SPEC entries are added to ct_extract_defs; 'adm:256' is added to dgcore.load_model (admwrap; fp16 unless DG_FP16=0);
#   * dgcore.to_t / ct_extract_defs.to_t: float16 arrays are the stored NHWC latents of enc_e512.py (passed through unchanged), uint8 arrays
#     are pixels as before;  * cfs_extract.py runs as cfs_extract_x.py with the probe in chunks of 4 (CFS_PROBE_CHUNK).
#   usage: python code/run_hires.py code/<script>.py <its arguments, with --tag in256c | e512>
import os, sys, runpy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, torch
import dgcore, ct_extract_defs
ct_extract_defs.SPEC['in256c'] = ('adm:256', 'hc_imagenet_bank', 8)
ct_extract_defs.SPEC['e512'] = ('edm2:edm2-img512-xs-uncond-2147483-0.045.pkl', 'lz512_bank_r0', 64)
ct_extract_defs.SPEC['dit256'] = ('dit:256', 'lzdit_bank_r0', 32)
_groups_for = ct_extract_defs.groups_for
def groups_for(net, den, bankset, dev):                                 # DiT: the probe modules of ditwrap.py
    return net.dg_groups if hasattr(net, 'dg_groups') else _groups_for(net, den, bankset, dev)
ct_extract_defs.groups_for = groups_for
_load_model = dgcore.load_model


def load_model(spec, device):
    if spec == 'dit:256':
        import ditwrap; return ditwrap.load_dit(device, fp16=os.environ.get('DG_FP16', '1') != '0')
    if spec == 'adm:256':
        import admwrap; return admwrap.load_adm(device, fp16=os.environ.get('DG_FP16', '1') != '0')
    return _load_model(spec, device)


_to_t = dgcore.to_t


def to_t(x, device):
    if getattr(x, 'dtype', None) == np.float16:                       # latents (enc_e512.py), NHWC
        return torch.from_numpy(np.ascontiguousarray(x)).to(device).permute(0, 3, 1, 2).float()
    return _to_t(x, device)


dgcore.load_model = load_model; dgcore.to_t = to_t; ct_extract_defs.to_t = to_t
script = sys.argv[1]
if os.path.basename(script) == 'cfs_extract.py':
    script = 'cfs_extract_x.py'; os.environ.setdefault('CFS_PROBE_CHUNK', '4')
sys.argv = [os.path.basename(script)] + sys.argv[2:]
runpy.run_path(os.path.join(HERE, os.path.basename(script)), run_name='__main__')
