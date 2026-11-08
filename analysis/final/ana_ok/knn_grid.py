# knn_grid.py -- what in the features and in the kNN head carries the gain of the tuned kNN over CFS (kNN): post hoc analysis, run on
#   the GPU server where the inputs are ($UPKNN_ROOT, default ~/upknn).  Nothing here changes any frozen choice.
#   (1) The 80 configurations of the tuned kNN (PROTOCOL_OK, extension 3: statistic P = R-th largest cosine / M = mean of the R largest;
#       blocks L1 / L012; noise sigma_1 (k0) / sigma_7 (k6); R in {1, ..., 1000}) from runs/ok/prof, evaluated on the common images of a
#       split: AUROC per pair (mean over noise seeds 0-2), natural-OOD mean per ID dataset, six-dataset mean, and the upside-down mean.
#   (2) Features vs head, CFS against our half-resolution block: the CFS descriptors (encoder + decoder 16x16 blocks, [mean, sqrt(var)],
#       logSNR 7, runs/cfs_l7) and our L1 features ([mean, std] of the last half-resolution decoder block, sigma_1, runs/ok/s*/k0), each
#       scored with the CFS head (standardised Euclidean distance, mean of the 10 nearest bank images) and with ours (cosine of the unit
#       vectors, 10th largest), plus the two mixed statistics; bank = the same 50,000 (ImageNet-64: 10,000) ID training images.
#   usage: UPKNN_ROOT=<root> python knn_grid.py <ours_vs_all_final2.json> <out.json>      (final2 and fresh splits; ~10 min on the CPU)
import os, sys, json, time
import numpy as np, torch
from scipy.stats import rankdata
torch.set_num_threads(8)
ROOT = os.environ.get('UPKNN_ROOT', os.path.expanduser('~/upknn'))
RGB = ['cifar10_test', 'cifar100_test', 'svhn_test', 'celeba_test', 'textures', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val']
TIERS = {
    'edmc10':   dict(id='cifar10_test', ood=[s for s in RGB if s != 'cifar10_test'] + ['cifar10_test__vflip', 'synth_noise', 'synth_const']),
    'l6svhn':   dict(id='svhn_test', ood=[s for s in RGB if s != 'svhn_test'] + ['svhn_test__vflip', 'synth_noise', 'synth_const']),
    'dpceleba': dict(id='celeba_test', ood=[s for s in RGB if s != 'celeba_test'] + ['celeba_test__vflip', 'synth_noise', 'synth_const']),
    'edmc100':  dict(id='cifar100_test', ood=[s for s in RGB if s != 'cifar100_test'] + ['cifar100_test__vflip', 'synth_noise', 'synth_const']),
    'in64':     dict(id='in64_imagenet_test', ood=['in64_ssb_hard', 'in64_ninco', 'in64_inaturalist', 'in64_textures', 'in64_openimage_o', 'in64_places365', 'in64_celeba_test', 'in64_imagenet_test__vflip']),
    'fmnist':   dict(id='g32_fmnist_test', ood=['g32_mnist_test', 'g32_kmnist_test', 'g32_omniglot_eval', 'g32_notmnist', 'g32_fmnist_test__vflip', 'g32_synth_noise', 'g32_synth_const']),
}
TAGS = list(TIERS); SEEDS = (0, 1, 2)
NAT = {t: [s for s in T['ood'] if not s.endswith('__vflip') and 'synth' not in s] for t, T in TIERS.items()}
VF = {t: [s for s in T['ood'] if s.endswith('__vflip')][0] for t, T in TIERS.items()}
NEAR = [('edmc10', 'cifar100_test'), ('edmc100', 'cifar10_test'), ('in64', 'in64_ssb_hard'), ('in64', 'in64_ninco')]
IDXDIR = {'final2': 'f2_idx', 'fresh': 'fresh_idx'}
REF = json.load(open(sys.argv[1])); OUTPATH = sys.argv[2]
EPS = 1e-6


def nm(tag, s): return TIERS[tag]['id'] if s == '__id__' else s
def comx(split, tag, s): return json.load(open(f'{ROOT}/eu/runs/jobs/{IDXDIR[split]}/{tag}__{nm(tag, s)}.json'))
def sub(score, idx, keep):
    pos = {int(i): j for j, i in enumerate(np.asarray(idx).tolist())}
    return np.asarray(score, np.float64)[[pos[i] for i in keep if i in pos]]
def auroc(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    r = rankdata(np.concatenate([a, b]))
    return float(100 * (r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def summarise(pairs):
    """pairs: {tag: {ood: auroc}} -> per-ID natural means, six-dataset mean, upside-down mean, near-OOD mean"""
    ids = {t: float(np.mean([pairs[t][o] for o in NAT[t]])) for t in TAGS}
    return dict(ids=ids, mean6=float(np.mean(list(ids.values()))), vflip=float(np.mean([pairs[t][VF[t]] for t in TAGS])),
                near=float(np.mean([pairs[t][o] for t, o in NEAR])), pairs=pairs)


# ---------------------------------------------------------------- (1) the 80 configurations
def grid(split):
    t0 = time.time(); acc = {}                                    # acc[cfg][tag][ood] = [auroc per seed]
    for tag in TAGS:
        keep_id = comx(split, tag, '__id__')
        for seed in SEEDS:
            def prof(s):
                n = f'{split}-{nm(tag, s)}'; Z = np.load(f'{ROOT}/runs/ok/prof/s{seed}/{tag}__{n}.npz')
                idx = np.load(f'{ROOT}/runs/ok/sims/s{seed}/{tag}/k0/{n}.npz')['idx']; return Z, idx
            Zi, ii = prof('__id__'); RS = list(Zi['R']); keys = [k for k in Zi.files if k != 'R']
            A = {(k, RS[j]): sub(Zi[k][:, j], ii, keep_id) for k in keys for j in range(len(RS))}
            for o in TIERS[tag]['ood']:
                Zo, io = prof(o); keep = comx(split, tag, o)
                for (k, R), a in A.items():
                    b = sub(Zo[k][:, RS.index(R)], io, keep)
                    acc.setdefault(f'{k}_r{R}', {}).setdefault(tag, {}).setdefault(o, []).append(auroc(a, b))
        print(f'  grid {split} {tag} done [{time.time() - t0:.0f}s]', flush=True)
    out = {}
    for cfg, P in acc.items():
        out[cfg] = summarise({t: {o: float(np.mean(v)) for o, v in P[t].items()} for t in TAGS})
    return out


# ---------------------------------------------------------------- (2) features vs head
def heads(split):
    t0 = time.time(); acc = {}
    for tag in TAGS:
        for seed in SEEDS:
            cd = f'{ROOT}/runs/cfs_l7/s{seed}/{tag}'; od = f'{ROOT}/runs/ok/s{seed}/{tag}/k0'
            feats = {}
            # CFS descriptors (raw [mean, sqrt var] of the two blocks)
            Bc = torch.from_numpy(np.load(f'{cd}/bank__z.npy')).float()
            feats['cfs'] = (Bc, lambda n: (torch.from_numpy(np.load(f'{cd}/{split}-{n}__z.npy')).float(), np.load(f'{cd}/{split}-{n}__idx.npy')))
            # our L1 features: [mean C, std C] of view 0 (the upright, +eps view)
            Xb = np.load(f'{od}/bank__L1.npy', mmap_mode='r'); C2 = Xb.shape[2] // 5
            Bo = torch.from_numpy(np.ascontiguousarray(Xb[:, 0, :C2])).float()
            feats['L1'] = (Bo, lambda n: (torch.from_numpy(np.ascontiguousarray(np.load(f'{od}/{split}-{n}__L1.npy', mmap_mode='r')[:, 0, :C2])).float(), np.load(f'{od}/{split}-{n}__idx.npy')))
            for fname, (Bank, load) in feats.items():
                mu, sd = Bank.mean(0), Bank.std(0, unbiased=False).clamp(min=EPS)
                Bs = (Bank - mu) / sd; Bs_sq = (Bs * Bs).sum(1); Bn = torch.nn.functional.normalize(Bank, dim=1)
                def scores(n):
                    Z, idx = load(n); out = {}
                    with torch.no_grad():
                        Zs = (Z - mu) / sd; d2 = ((Zs * Zs).sum(1, keepdim=True) + Bs_sq[None] - 2 * Zs @ Bs.T).clamp(min=0)
                        d10 = d2.topk(10, dim=1, largest=False).values.sqrt()
                        out['std_mean10'] = -d10.mean(1).numpy(); out['std_10th'] = -d10[:, 9].numpy()
                        c10 = (torch.nn.functional.normalize(Z, dim=1) @ Bn.T).topk(10, dim=1).values
                        out['cos_10th'] = c10[:, 9].numpy(); out['cos_mean10'] = c10.mean(1).numpy()
                    return out, idx
                Si, ii = scores(nm(tag, '__id__')); keep_id = comx(split, tag, '__id__')
                A = {h: sub(v, ii, keep_id) for h, v in Si.items()}
                for o in TIERS[tag]['ood']:
                    So, io = scores(nm(tag, o)); keep = comx(split, tag, o)
                    for h in A:
                        acc.setdefault(f'{fname}_{h}', {}).setdefault(tag, {}).setdefault(o, []).append(auroc(A[h], sub(So[h], io, keep)))
        print(f'  heads {split} {tag} done [{time.time() - t0:.0f}s]', flush=True)
    return {k: summarise({t: {o: float(np.mean(v)) for o, v in P[t].items()} for t in TAGS}) for k, P in acc.items()}


OUT = {'grid': {}, 'heads': {}, 'check': {}}
for split in ('final2', 'fresh'):
    OUT['grid'][split] = grid(split); OUT['heads'][split] = heads(split)
# checks against the frozen rows of the paper (final images): kNN (CT) = P_L1_k0 at R = 10; CFS (kNN) = the CFS head on the CFS descriptors
g = OUT['grid']['final2']['P_L1_k0_r10']['ids']; h = OUT['heads']['final2']
ct = {t: REF['ids'][t]['rows']['kNN (CT)']['mean'] for t in TAGS}; cf = {t: REF['ids'][t]['rows']['CFS-kNN']['mean'] for t in TAGS}
OUT['check'] = {'knn_ct_maxabs': max(abs(g[t] - ct[t]) for t in TAGS), 'cfs_knn_maxabs': max(abs(h['cfs_std_mean10']['ids'][t] - cf[t]) for t in TAGS),
                'L1_cos_10th_vs_grid_maxabs': max(abs(h['L1_cos_10th']['ids'][t] - g[t]) for t in TAGS)}
print('checks', OUT['check'], flush=True)
json.dump(OUT, open(OUTPATH, 'w'), indent=1)
print('saved', OUTPATH, flush=True)
