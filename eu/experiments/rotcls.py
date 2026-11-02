# rotcls.py -- learned rotation classifiers of one ID training set (ResNet-18), the trained counterpart of Euphonium's ratio
#   kind bin90 : upright training images (label 1) vs R270-rotated training images (label 0).  If x ~ p is upright and z = R270 y,
#                y ~ p, then z has the density q(z) = p(R90 z), so the Bayes log-odds is log p(x) - log p(R90 x) = Lambda_90(x),
#                the quantity whose noise-prediction estimate Euphonium's R90 atoms read.
#   kind bin180: upright vs R180-rotated images; Bayes log-odds = log p(x) - log p(R180 x) = Lambda_180(x).
#   kind rot4  : four-way rotation prediction (Golan & El-Yaniv 2018; Hendrycks et al. 2019): every image in all four rotations,
#                score of a test image = sum_r log p(r | R_r x) (higher = more in-distribution).
#   Architecture: torchvision ResNet-18 with the CIFAR stem (3x3 conv, stride 1, no max-pool).  SGD (Nesterov, momentum 0.9,
#   weight decay 5e-4), batch 256 (rot4: 128 images x 4 rotations), cosine learning rate from 0.1, mixed precision.
#   Augmentation (applied before the rotation): random crop with reflect padding of 1/8 of the side, horizontal flip for
#   natural-image sets (--hflip 1; off for digits and clothing, whose mirror images are not in the data).
#   A fixed 5000-image hold-out of the training set (permutation seed 1234, the calibration split of train.py) is used only to
#   report the validation accuracy; the last epoch is kept.
#   usage: python eu/experiments/rotcls.py --data cifar10_train --kind bin90 --out runs/rotcls/edmc10/bin90 --gpu 0 [--epochs 30]
import os, sys, time, json, math, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rotcls_net import build

ROOT = os.environ.get('SR_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
NPY = f'{ROOT}/data/npy'
p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--kind', required=True, choices=['bin90', 'bin180', 'rot4'])
p.add_argument('--out', required=True)
p.add_argument('--epochs', type=int, default=30)
p.add_argument('--bs', type=int, default=256)
p.add_argument('--lr', type=float, default=0.1)
p.add_argument('--wd', type=float, default=5e-4)
p.add_argument('--hflip', type=int, default=1)
p.add_argument('--hold', type=int, default=5000)
p.add_argument('--extra', default='', help='comma-separated extra training arrays (e.g. additional ImageNet-64 training images)')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--gpu', type=int, default=0)
a = p.parse_args()
out = a.out if os.path.isabs(a.out) else f'{ROOT}/{a.out}'
if os.path.exists(f'{out}/final.pt'):
    print('exists', out); sys.exit(0)
os.makedirs(out, exist_ok=True)
dev = torch.device(f'cuda:{a.gpu}')
torch.manual_seed(a.seed); np.random.seed(a.seed); torch.backends.cudnn.benchmark = True


ROT = lambda x, r: x.rot90(r, (2, 3))
if __name__ == '__main__':
    x_all = np.load(f'{NPY}/{a.data}.npy')
    if x_all.ndim == 3: x_all = x_all[..., None]
    perm = np.random.RandomState(1234).permutation(len(x_all))
    hold_idx, train_idx = np.sort(perm[:a.hold]), np.sort(perm[a.hold:])
    xs = [x_all[train_idx]]
    for e in [s for s in a.extra.split(',') if s]:
        z = np.load(f'{NPY}/{e}.npy'); xs.append(z[..., None] if z.ndim == 3 else z)
    data = torch.from_numpy(np.concatenate(xs)).to(dev)               # uint8 [N, H, W, C] on the GPU
    hold = torch.from_numpy(x_all[hold_idx]).to(dev)
    N, H, W, C = data.shape
    pad = H // 8
    net = build(C, 1 if a.kind != 'rot4' else 4).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(net.parameters(), lr=a.lr, momentum=0.9, nesterov=True, weight_decay=a.wd)
    scaler = torch.amp.GradScaler('cuda')
    ipb = a.bs if a.kind != 'rot4' else a.bs // 2                        # images per batch (rot4: x4 rotations)
    steps_per_epoch = N // ipb
    total = steps_per_epoch * a.epochs

    def to_f(u): return u.permute(0, 3, 1, 2).float() / 127.5 - 1.0

    def augment(x):
        B = x.shape[0]
        xp = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        ox = torch.randint(0, 2 * pad + 1, (B,), device=dev); oy = torch.randint(0, 2 * pad + 1, (B,), device=dev)
        ar = torch.arange(H, device=dev)
        rows = (oy[:, None] + ar[None]).view(B, 1, H, 1).expand(B, C, H, W + 2 * pad)
        x = torch.gather(xp, 2, rows)
        cols = (ox[:, None] + ar[None]).view(B, 1, 1, W).expand(B, C, H, W)
        x = torch.gather(x, 3, cols)
        if a.hflip: x = torch.where(torch.rand(B, 1, 1, 1, device=dev) < 0.5, x.flip(3), x)
        return x

    def make_batch(x):
        """x: augmented upright images -> (inputs, targets)"""
        if a.kind == 'rot4':
            xs = torch.cat([ROT(x, r) for r in range(4)])
            return xs, torch.arange(4, device=dev).repeat_interleave(x.shape[0])
        neg = torch.rand(x.shape[0], device=dev) < 0.5
        r = 3 if a.kind == 'bin90' else 2                                 # negatives: R270 (bin90) or R180 (bin180) images
        xs = torch.where(neg.view(-1, 1, 1, 1), ROT(x, r), x)
        return xs, (~neg).float()

    @torch.no_grad()
    def evaluate():
        net.eval(); acc, nll, n = 0.0, 0.0, 0
        for i in range(0, len(hold), 500):
            x = to_f(hold[i:i + 500])
            if a.kind == 'rot4':
                xs = torch.cat([ROT(x, r) for r in range(4)]); y = torch.arange(4, device=dev).repeat_interleave(x.shape[0])
                with torch.autocast('cuda', dtype=torch.float16): lo = net(xs.contiguous(memory_format=torch.channels_last)).float()
                acc += (lo.argmax(1) == y).float().sum().item(); nll += F.cross_entropy(lo, y, reduction='sum').item(); n += len(y)
            else:
                r = 3 if a.kind == 'bin90' else 2
                xs = torch.cat([x, ROT(x, r)]); y = torch.cat([torch.ones(len(x), device=dev), torch.zeros(len(x), device=dev)])
                with torch.autocast('cuda', dtype=torch.float16): lo = net(xs.contiguous(memory_format=torch.channels_last)).float()[:, 0]
                acc += ((lo > 0).float() == y).float().sum().item()
                nll += F.binary_cross_entropy_with_logits(lo, y, reduction='sum').item(); n += len(y)
        net.train()
        return acc / n, nll / n

    print(f'{a.data} {a.kind} train {N} hold {len(hold)} res {H} ch {C} steps {total} (epochs {a.epochs})', flush=True)
    log, t0, step = [], time.time(), 0
    net.train()
    for ep in range(a.epochs):
        order = torch.randperm(N, device=dev)
        la, ln = 0.0, 0
        for it in range(steps_per_epoch):
            lr = a.lr * 0.5 * (1 + math.cos(math.pi * step / total))
            for g in opt.param_groups: g['lr'] = lr
            x = augment(to_f(data[order[it * ipb:(it + 1) * ipb]]))
            xs, y = make_batch(x)
            with torch.autocast('cuda', dtype=torch.float16):
                lo = net(xs.contiguous(memory_format=torch.channels_last)).float()
            loss = F.cross_entropy(lo, y) if a.kind == 'rot4' else F.binary_cross_entropy_with_logits(lo[:, 0], y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            la += float(loss.detach()); ln += 1; step += 1
        acc, nll = evaluate()
        log.append(dict(epoch=ep + 1, train_loss=la / ln, hold_acc=acc, hold_nll=nll, sec=time.time() - t0))
        print(f'epoch {ep + 1} loss {la / ln:.4f} hold acc {acc:.4f} nll {nll:.4f} {time.time() - t0:.0f}s', flush=True)
    torch.save(dict(net=net.state_dict(), args=vars(a), ch=C, res=H, n_out=1 if a.kind != 'rot4' else 4, log=log,
                    train_n=N, hold_idx=hold_idx), f'{out}/final.pt')
    json.dump(dict(args=vars(a), log=log, train_n=N), open(f'{out}/log.json', 'w'), indent=1)
    print('TRAIN_DONE', out, flush=True)
