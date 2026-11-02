# run_adm.py -- run any Dragon Glider script with the ADM ImageNet-256 model (admwrap.py) as '--model adm:256' (Experiment A, 256x256).
#   usage: python code/run_adm.py [--kidx 0,2,3,5,7] [--idxjson f --idxkey tier:set] code/<script>.py <its arguments>
#   * load_model('adm:256') is added (dgcore.load_model patched before the script is run).
#   * --kidx (extract2.py only): rails on a subset of the 16-scale grid with the shared-noise draws of the full grid (as extract_bu.py);
#     the target scales are replaced by the model's nearest sigma_t (admwrap.sigmas_for), so that eps_hat is exactly the model's output.
#   * --idxjson/--idxkey: explicit image indices of --data (as extract_bu.py).
import os, sys, json, runpy
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import dgcore, admwrap

args = sys.argv[1:]


def take(flag):
    if flag not in args: return None
    i = args.index(flag); v = args[i + 1]; del args[i:i + 2]; return v


kidx = take('--kidx'); idxjson, idxkey = take('--idxjson'), take('--idxkey')
script = args.pop(0)
_load_model = dgcore.load_model
def load_model(spec, device):
    if spec == 'adm:256': return admwrap.load_adm(device)
    return _load_model(spec, device)
dgcore.load_model = load_model
if kidx:
    kidx = [int(k) for k in kidx.split(',')]
    S16 = np.exp(np.linspace(np.log(0.01), np.log(31.6228), 16))
    sig = admwrap.sigmas_for([S16[k] for k in kidx])
    _shared_noise = dgcore.shared_noise
    def shared_noise(shape, seed, k, j, device): return _shared_noise(shape, seed, kidx[k], j, device)
    dgcore.shared_noise = shared_noise
    args += ['--sigmas', ','.join('%.8g' % s for s in sig)]
    print('[run_adm] scales', ' '.join('%.4g' % s for s in sig), '(targets', ' '.join('%.4g' % S16[k] for k in kidx) + ')', flush=True)
if idxjson:
    tier, name = idxkey.split(':', 1)
    IDX = np.array(sorted(int(i) for i in json.load(open(idxjson))[tier][name]), dtype=np.int64)
    def select(name, n=None, seed=0, complement=False, cap=0):
        full = np.load(f'{dgcore.NPY}/{name}.npy', mmap_mode='r'); return np.ascontiguousarray(full[IDX]), IDX
    dgcore.select = select
sys.argv = [os.path.basename(script)] + args
runpy.run_path(os.path.join(HERE, os.path.basename(script)), run_name='__main__')
