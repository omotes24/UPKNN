# mk_q_6id.py -- appends to runs/jobs/Q_hires.txt (after the hires and DiT jobs) the three further comparison methods of the six ID tiers,
#   run on the FINAL images of Upright kNN (final2-* rows of runs/ok/sets_<tag>.json, 500 per set): DDPM-OOD of Graham et al. (graham.py),
#   LMD (lmd.py) and the likelihood + LID method of Kamkari et al. (kamkari_lid.py; not for ImageNet-64, too expensive as before), with
#   the settings of their earlier runs (eu/jobs); outputs under runs/<method>/<tag>_f2/ (the earlier 128-image runs stay untouched); the
#   calibration runs of the earlier runs (Graham: 128 ID development images; Kamkari: 128 training images) are linked, not recomputed.
#   Added after the frozen evaluation as comparison rows (like CFS and the rotation classifier).
import os, json
ROOT = os.path.expanduser("~/upknn"); os.chdir(ROOT); Q = "runs/jobs/Q_hires.txt"; JD = "eu/runs/jobs/f2_idx"; os.makedirs(JD, exist_ok=True)
SPEC = {"edmc10": ("edm:edm-cifar10-32x32-uncond-vp.pkl", "", "cifar10_test", "cifar10_train"), "l6svhn": ("l6:layer6_svhn_vpsde.pt", "", "svhn_test", "svhn_train"),
        "dpceleba": ("iddpm:diffpath_celeba32.pt", "", "celeba_test", "celeba_train"), "edmc100": ("edmsd:edm_cifar100_uncond_vp.pt", "SR_FP16=0 ", "cifar100_test", "cifar100_train"),
        "in64": ("edm2:edm2-img64-xs-uncond-2147483-0.100.pkl", "", "in64_imagenet_test", None), "fmnist": ("own:models/fmnist_s/final.pt", "", "g32_fmnist_test", "g32_fmnist_train")}
BS = {"graham": {"edmc100": 32, "in64": 32}, "lmd": {"edmc100": 25, "in64": 10}}
E = "SR_ROOT=$HOME/upknn "
lines = []
for meth in ("graham", "lmd", "kamkari_lid"):
    for t, (spec, env, idn, trn) in SPEC.items():
        if meth == "kamkari_lid" and t == "in64": continue
        mdir = meth.split("_")[0]; od = f"runs/{mdir}/{t}_f2"; os.makedirs(od, exist_ok=True)
        if meth == "graham":
            src = f"runs/graham/{t}/{idn}__dev128.npz"; dst = f"{od}/{idn}__dev128.npz"
            if not os.path.lexists(dst): os.symlink(os.path.realpath(src), dst)
        if meth == "kamkari_lid":
            src = f"runs/kamkari/{t}/{trn}__k128.npz"; dst = f"{od}/{trn}__k128.npz"
            if not os.path.lexists(dst): os.symlink(os.path.realpath(src), dst)
        for name, data, idx, pre in json.load(open(f"runs/ok/sets_{t}.json")):
            if not name.startswith("final2-"): continue
            nm = name[len("final2-"):]
            assert (pre == "none" and nm == data) or (pre == "vflip" and nm == data + "__vflip"), (t, name, data, pre)
            jp = f"{JD}/{t}__{nm}.json"; json.dump([int(i) for i in idx], open(jp, "w"))
            if meth == "kamkari_lid":
                ch = 256 if t == "edmc100" else 512; opt = f"--chunk {ch}"
            else:
                bs = BS[meth].get(t, 64 if meth == "graham" else 50); opt = f"--bs {bs}"
            lines.append(f"{E}{env}python eu/experiments/{meth}.py --gpu {{gpu}} --model {spec} --tag {t}_f2 --sets {nm}:{jp} {opt} > logs/hires/{meth}_{t}_{nm}.log 2>&1")
open(Q, "a").write("\n".join(lines) + "\n"); print(len(lines), "jobs appended")
