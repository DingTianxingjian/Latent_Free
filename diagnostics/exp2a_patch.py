"""Item 2 -- Exp 2A in patch space, plus a per-condition null control.

Two purposes in one job:

(A) The distractor table (Exp 2A) was the last headline still measured only on
    the CLS token.  Re-run the within-condition probe set on the 4x4 patch grid
    (single / pair_shuf / pair / Delta-f), for all five conditions.

(B) The variance!=action test on clean data is nearly circular: in clean LIBERO
    the ONLY thing moving is the robot, so of course its motion dominates both
    variance and action and top-k wins.  The honest test needs *exogenous*
    variance.  The moving distractor supplies it.  So per condition we redo the
    top-k / random-k / bottom-k null (as in exp2b_null) and ask whether the
    moving distractor narrows or reverses top-k's advantage over random-k.

    Verdict for (B): under `patch` (moving) the top-k/random-k edge shrinks or
    inverts -> "variance-greedy selection succeeds or fails depending on who owns
    the variance" (a conditional mechanism survives).  Unchanged -> the
    variance story is dead even with a distractor present.

Space: g4 (12288-d), where the action signal lives (Exp 4).  Part A uses the GPU
ridge (probes_torch) for the wide pair features; Part B refits inside k<=128
projections with the same sklearn probe as exp2b_null.  Gripper is omitted from
Part A (binary, not the transition question); continuous R^2 only.
"""

import os
import time

import numpy as np
import pandas as pd
import torch

import data
import probes
import probes_torch
from features import CONDITIONS

SPACE, DIM = "g4", 12288
KS = (8, 16, 32, 64, 128)
N_SEED = 5


def load(cond):
    p = os.path.join(data.CACHE_DIR, f"pfeats_{cond}_{SPACE}.npy")
    return np.load(p) if os.path.exists(p) else None


def part_a(cond, feats, index, dev):
    rows = []
    for stride in data.STRIDES:
        itr, jtr, yc_tr, _ = data.build_pairs(index, stride, "train")
        iva, jva, yc_va, _ = data.build_pairs(index, stride, "val")
        rng = np.random.RandomState(stride)
        ptr, pva = rng.permutation(len(jtr)), rng.permutation(len(jva))

        def build(kind):                        # build one feature pair lazily
            if kind == "single":
                return feats[itr], feats[iva]
            if kind == "diff":
                return feats[jtr] - feats[itr], feats[jva] - feats[iva]
            if kind == "pair":
                return (np.concatenate([feats[itr], feats[jtr]], 1),
                        np.concatenate([feats[iva], feats[jva]], 1))
            if kind == "pair_shuf":
                return (np.concatenate([feats[itr], feats[jtr][ptr]], 1),
                        np.concatenate([feats[iva], feats[jva][pva]], 1))

        for kind in ("single", "diff", "pair", "pair_shuf"):
            if dev == "cuda":
                torch.cuda.empty_cache()
            Ftr, Fva = build(kind)
            r, _ = probes_torch.ridge_probe_t(Ftr, yc_tr, Fva, yc_va, device=dev)
            rows.append(dict(condition=cond, stride=stride, part="A", feature=kind,
                             dim=Ftr.shape[1], r2_xyz=r["r2_xyz"], r2_rot=r["r2_rot"],
                             r2_cont6=r["r2_cont6"], alpha=r["alpha"]))
            print(f"[2a:A] {cond:14s} s={stride} {kind:9s} d={Ftr.shape[1]:5d}  "
                  f"xyz={r['r2_xyz']:.3f} rot={r['r2_rot']:.3f} "
                  f"cont6={r['r2_cont6']:.3f}", flush=True)
            del Ftr, Fva
    return rows


def part_b(cond, feats, index, dev):
    rows = []
    for stride in data.STRIDES:
        itr, jtr, yc_tr, _ = data.build_pairs(index, stride, "train")
        iva, jva, yc_va, _ = data.build_pairs(index, stride, "val")
        Dtr = (feats[jtr] - feats[itr]).astype(np.float32)
        Dva = (feats[jva] - feats[iva]).astype(np.float32)
        full, _ = probes.ridge_probe(Dtr, yc_tr, Dva, yc_va)
        xyz_full = full["r2_xyz"]

        Gtr = torch.from_numpy(Dtr).to(dev)
        Gva = torch.from_numpy(Dva).to(dev)
        _, _, Vh = torch.linalg.svd(Gtr - Gtr.mean(0, keepdim=True), full_matrices=False)

        def pr(Q):
            Ztr, Zva = (Gtr @ Q).cpu().numpy(), (Gva @ Q).cpu().numpy()
            res, _ = probes.ridge_probe(Ztr, yc_tr, Zva, yc_va)
            return res["r2_xyz"]

        for k in KS:
            top = pr(Vh[:k].T)
            bot = pr(Vh[-k:].T)
            rand = [pr(torch.linalg.qr(torch.randn(
                DIM, k, generator=torch.Generator(dev).manual_seed(s), device=dev))[0])
                for s in range(N_SEED)]
            rand = np.array(rand)
            rows.append(dict(condition=cond, stride=stride, part="B", k=k, D=DIM,
                             r2_xyz_full=xyz_full, r2_xyz_top=top,
                             r2_xyz_rand_mean=float(rand.mean()),
                             r2_xyz_rand_std=float(rand.std()), r2_xyz_bottom=bot,
                             top_over_rand=top / rand.mean() if rand.mean() > 1e-9 else float("nan")))
            print(f"[2a:B] {cond:14s} s={stride} k={k:3d}  top={top:.3f} "
                  f"rand={rand.mean():.3f}±{rand.std():.3f} bottom={bot:.3f}  "
                  f"top/rand={rows[-1]['top_over_rand']:.2f}x", flush=True)
        del Gtr, Gva, Vh, Dtr, Dva
        if dev == "cuda":
            torch.cuda.empty_cache()
    return rows


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"[2a] GPU {torch.cuda.get_device_name(0)}  "
              f"free {free/2**30:.1f} / {total/2**30:.1f} GiB", flush=True)
    print(f"[2a] device={dev}  space={SPACE}", flush=True)
    index = data.build_index()
    A, B = [], []
    for cond in CONDITIONS:
        feats = load(cond)
        if feats is None:
            print(f"[2a] {cond}: patch features missing -- run features_patch first")
            continue
        t0 = time.time()
        A += part_a(cond, feats, index, dev)
        B += part_b(cond, feats, index, dev)
        print(f"[2a] {cond} done [{time.time()-t0:.0f}s]", flush=True)
        del feats

    pd.DataFrame(A).to_csv(os.path.join(data.RESULTS_DIR, "exp2a_patch_probes.csv"), index=False)
    pd.DataFrame(B).to_csv(os.path.join(data.RESULTS_DIR, "exp2a_patch_null.csv"), index=False)
    print("[2a] wrote results/exp2a_patch_probes.csv and results/exp2a_patch_null.csv")


if __name__ == "__main__":
    main()
