# csi_run.py -- CSI (Tack et al., NeurIPS 2020; official code ext/CSI = alinlab/CSI @ 60742b6, unmodified) on the ID tiers of the paper
#   (runs/kx/PROTOCOL_CSI.md).  This wrapper only
#   (1) feeds our arrays to CSI instead of torchvision data sets (training images = the bank of the tier, ct_extract_defs.bank_indices;
#       3 channels, float in [0, 1] as torchvision ToTensor);
#   (2) provides `torchlars.LARS` (the semantics of torchlars 0.1.2, which does not build with our torch), a stub of `diffdist`
#       (used only by multi-GPU runs) and of `tensorboardX.SummaryWriter` (not installed; scalars go to <log dir>/scalars.tsv);
#   (2b) passes the first two entries of image_size = (32, 32, 3) as the output size of RandomResizedCropLayer's adaptive_avg_pool2d:
#       torch < 1.8 ignored the third entry (the pooling is then the identity at 32x32), newer torch rejects it;
#   (3) lets an interrupted run resume: utils.load_checkpoint reads config['best'], which the unsupervised modes never save (default
#       100.0 as on a fresh start); a log directory without a checkpoint is renamed, never deleted; no interactive prompt;
#   (4) for scoring, applies the official 'CSI' score of evals/ood_pre.py (same functions; axis and weights from the training images,
#       README flags --ood_samples 10 --resize_factor 0.54 --resize_fix) to our evaluation images and stores per-image scores.
#   train:  python code/csi_run.py train <tag> <gpu> [epochs=1000] [batch=128]      (resumes from runs/csi/logs/<run>/last.model)
#   score:  python code/csi_run.py score <tag> <gpu> [prefixes=final2-]               -> runs/csi/scores/<tag>/<set>.npz {score, idx}
#   CSI_SMOKE=<n>: use only the first n bank images (smoke tests);  CSI_RUN=<dir>: run directory (default runs/csi)
import os, sys, types, json, time, runpy, builtins
import numpy as np, torch
from torch.optim import Optimizer
ROOT = os.path.expanduser('~/upknn'); CSI = f'{ROOT}/ext/CSI'
RUN = os.path.abspath(os.path.expanduser(os.environ.get('CSI_RUN', f'{ROOT}/runs/csi')))
mode, tag, gpu = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ['CUDA_VISIBLE_DEVICES'] = gpu                       # CSI uses torch.cuda.device_count(): one visible GPU -> single-GPU code path


def _no_input(*a, **k): raise RuntimeError('csi_run.py: interactive input requested (non-empty log dir without --resume_path?)')
builtins.input = _no_input


class LARS(Optimizer):
    """torchlars.LARS (v0.1.2): layer-wise adaptive rate scaling around a base optimizer; weight decay is folded into the gradient
    before the trust ratio trust_coef * |w| / (|g| + wd |w| + eps) is applied, and hidden from the base optimizer during its step."""
    def __init__(self, optimizer, eps=1e-8, trust_coef=0.001):
        self.optim, self.eps, self.trust_coef = optimizer, eps, trust_coef
    @property
    def param_groups(self): return self.optim.param_groups
    @property
    def state(self): return self.optim.state
    @property
    def defaults(self): return self.optim.defaults
    def state_dict(self): return self.optim.state_dict()
    def load_state_dict(self, sd): return self.optim.load_state_dict(sd)
    def zero_grad(self, set_to_none=True): return self.optim.zero_grad()
    def add_param_group(self, g): return self.optim.add_param_group(g)
    def __repr__(self): return f'LARS({self.optim!r})'
    def step(self, closure=None):
        wds = [g.get('weight_decay', 0.0) for g in self.optim.param_groups]
        for g in self.optim.param_groups: g['weight_decay'] = 0.0
        try:
            with torch.no_grad():
                for g, wd in zip(self.optim.param_groups, wds):
                    for p in g['params']:
                        if p.grad is None: continue
                        pn, gn = p.norm(), p.grad.norm()
                        r = torch.where((pn > 0) & (gn > 0), self.trust_coef * pn / (gn + wd * pn + self.eps), torch.ones_like(pn))
                        p.grad.add_(p, alpha=wd); p.grad.mul_(r)
                return self.optim.step(closure)
        finally:
            for g, wd in zip(self.optim.param_groups, wds): g['weight_decay'] = wd


class SummaryWriter:
    def __init__(self, logdir, *a, **k): self.f = open(os.path.join(logdir, 'scalars.tsv'), 'a')
    def add_scalar(self, tag, value, step, *a, **k): self.f.write(f'{tag}\t{step}\t{float(value)!r}\n'); self.f.flush()
    def add_image(self, *a, **k): pass
    def add_histogram(self, *a, **k): pass
    def flush(self): self.f.flush()
    def close(self): self.f.close()


tb = types.ModuleType('tensorboardX'); tb.SummaryWriter = SummaryWriter; sys.modules['tensorboardX'] = tb
tl = types.ModuleType('torchlars'); tl.LARS = LARS; sys.modules['torchlars'] = tl
dd = types.ModuleType('diffdist'); ddf = types.ModuleType('diffdist.functional')
def _no_multi_gpu(*a, **k): raise RuntimeError('diffdist is only used by multi-GPU CSI runs')
ddf.all_gather = _no_multi_gpu; dd.functional = ddf; sys.modules['diffdist'] = dd; sys.modules['diffdist.functional'] = ddf
sys.path.insert(0, CSI); sys.path.append(f'{ROOT}/code')       # CSI's packages (common, datasets, models, utils, ...) first
from ct_extract_defs import SPEC, bank_indices
NPY = f'{ROOT}/data/npy'


class Arr(torch.utils.data.Dataset):
    def __init__(self, name, idx, pre='none'):
        self.a = np.load(f'{NPY}/{name}.npy', mmap_mode='r'); self.idx = np.asarray(idx); self.pre = pre
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        x = np.array(self.a[int(self.idx[i])])                     # a writable copy of the uint8 image
        if x.ndim == 2: x = np.repeat(x[..., None], 3, 2)
        if self.pre == 'vflip': x = x[::-1]
        elif self.pre != 'none': raise ValueError(self.pre)
        return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float() / 255., 0


def bank_set():
    bankset = SPEC[tag][1]; n = len(np.load(f'{NPY}/{bankset}.npy', mmap_mode='r')); bidx, _ = bank_indices(n)
    if os.environ.get('CSI_SMOKE'): bidx = bidx[:int(os.environ['CSI_SMOKE'])]
    return Arr(bankset, bidx)


def get_dataset(P, dataset, test_only=False, image_size=None, download=False, eval=False):
    assert not test_only, 'no torchvision OOD sets are used (--ood_dataset is empty)'
    tr = bank_set(); return tr, Arr(SPEC[tag][1], tr.idx[:100]), (32, 32, 3), 10      # test set: unused by the unsupervised modes


import datasets as CSI_DATASETS
assert os.path.dirname(CSI_DATASETS.__file__).startswith(CSI), CSI_DATASETS.__file__
CSI_DATASETS.get_dataset = get_dataset
import models.transform_layers as TL
_rrc_init = TL.RandomResizedCropLayer.__init__
def _rrc_init_2d(self, size=None, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
    _rrc_init(self, size=None if size is None else tuple(size)[:2], scale=scale, ratio=ratio)
TL.RandomResizedCropLayer.__init__ = _rrc_init_2d
import utils.utils as CSI_UTILS
_load_checkpoint = CSI_UTILS.load_checkpoint
def load_checkpoint(logdir, mode='last'):
    m, o, cfg = _load_checkpoint(logdir, mode)
    if cfg is not None: cfg.setdefault('best', 100.0)
    return m, o, cfg
CSI_UTILS.load_checkpoint = load_checkpoint
os.makedirs(RUN, exist_ok=True); os.chdir(RUN)
COMMON = ['--dataset', 'cifar10', '--model', 'resnet18', '--shift_trans_type', 'rotation', '--suffix', tag, '--ood_dataset']
FNAME = f'cifar10_resnet18_unsup_simclr_CSI_shift_rotation_{tag}'
LOG = f'{RUN}/logs/{FNAME}'

if mode == 'train':
    epochs = sys.argv[4] if len(sys.argv) > 4 else '1000'; bs = sys.argv[5] if len(sys.argv) > 5 else '128'
    if os.path.exists(f'{LOG}/TRAIN_DONE'):
        print('CSI train', tag, 'already done:', open(f'{LOG}/TRAIN_DONE').read().strip(), flush=True); sys.exit(0)
    args = ['train.py', '--mode', 'simclr_CSI', '--epochs', epochs, '--batch_size', bs, '--save_step', '10'] + COMMON
    if os.path.exists(f'{LOG}/last.model'): args[1:1] = ['--resume_path', f'logs/{FNAME}']
    elif os.path.isdir(LOG) and os.listdir(LOG):
        os.rename(LOG, f'{LOG}.incomplete_{time.strftime("%Y%m%d_%H%M%S")}')
    sys.argv = args; print('CSI train', tag, args, 'bank', len(bank_set()), flush=True); t0 = time.time()
    runpy.run_path(f'{CSI}/train.py', run_name='__main__')
    gb = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f'CSI train {tag} finished: {(time.time() - t0) / 3600:.2f} h this run, peak {gb:.2f} GiB', flush=True)
    open(f'{LOG}/TRAIN_DONE', 'w').write(f'{epochs} epochs, batch {bs}, bank {len(bank_set())}, {time.strftime("%F %T")}\n')

elif mode == 'score':
    prefixes = tuple((sys.argv[4] if len(sys.argv) > 4 else 'final2-').split(','))
    assert os.path.exists(f'{LOG}/TRAIN_DONE') or os.environ.get('CSI_SMOKE'), 'training not finished'
    sys.argv = ['eval.py', '--mode', 'ood_pre', '--ood_score', 'CSI', '--ood_samples', '10', '--resize_factor', '0.54', '--resize_fix',
                '--load_path', f'logs/{FNAME}/last.model'] + COMMON
    from common.common import parse_args
    import models.classifier as C
    from evals.ood_pre import get_features, get_scores
    from utils.utils import normalize
    P = parse_args(); P.n_gpus = 1; P.multi_gpu = False; P.image_size = (32, 32, 3); P.n_classes = 10   # == common/eval.py
    dev = torch.device('cuda')
    simclr_aug = C.get_simclr_augmentation(P, image_size=P.image_size).to(dev)
    P.shift_trans, P.K_shift = C.get_shift_module(P, eval=True); P.shift_trans = P.shift_trans.to(dev)
    model = C.get_classifier(P.model, n_classes=P.n_classes).to(dev); model = C.get_shift_classifer(model, P.K_shift).to(dev)
    model.load_state_dict(torch.load(P.load_path, map_location=dev)); model.eval()
    kw = {'pin_memory': False, 'num_workers': 4}
    train_loader = torch.utils.data.DataLoader(bank_set(), shuffle=True, batch_size=P.batch_size, **kw)
    prefix = os.path.join(os.path.split(P.load_path)[0], f'feats_{P.ood_samples}_resize_fix_{P.resize_factor}')
    fa = dict(simclr_aug=simclr_aug, sample_num=P.ood_samples, layers=P.ood_layer)
    od = f'{RUN}/scores/{tag}'; os.makedirs(od, exist_ok=True); t0 = time.time()
    with torch.no_grad():                                         # == evals/ood_pre.py: eval_ood_detection, 'CSI' score
        ft = get_features(P, f'{tag}_train', model, train_loader, prefix=prefix, **fa)
        P.axis = [normalize(f.mean(dim=1), dim=1).to(dev) for f in ft['simclr'].chunk(P.K_shift, dim=1)]
        f_sim = [f.mean(dim=1) for f in ft['simclr'].chunk(P.K_shift, dim=1)]; f_shi = [f.mean(dim=1) for f in ft['shift'].chunk(P.K_shift, dim=1)]
        P.weight_sim = [1 / f_sim[k].norm(dim=1).mean().item() for k in range(P.K_shift)]
        P.weight_shi = [1 / f_shi[k][:, k].mean().item() for k in range(P.K_shift)]
        print('axis', [len(a) for a in P.axis], 'weight_sim', P.weight_sim, 'weight_shi', P.weight_shi, flush=True)
        for name, data, idx, pre in json.load(open(f'{ROOT}/runs/ok/sets_{tag}.json')):
            if not name.startswith(prefixes): continue
            ld = torch.utils.data.DataLoader(Arr(data, idx, pre), shuffle=False, batch_size=P.test_batch_size, **kw)
            sc = get_scores(P, get_features(P, name, model, ld, prefix=prefix, **fa), P.ood_score[0]).numpy()
            np.savez(f'{od}/{name}.npz', score=sc.astype(np.float64), idx=np.asarray(idx)); print(tag, name, len(sc), float(sc.mean()), flush=True)
    open(f'{od}/SCORE_DONE', 'w').write(f'{",".join(prefixes)} {time.time() - t0:.0f} s {time.strftime("%F %T")}\n')
