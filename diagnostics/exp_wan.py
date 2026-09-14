"""Wan2.2 VAE space -- the same protocol as DINOv2, for cross-encoder comparison.

Answers item-3 predictions on a reconstruction-target space:
  (1) state confusion holds       -> pair_shuf ~= single here too?
  (2) lower per-dim decodability   -> single/Δf R^2 vs DINOv2 pg4?
  (3) narrower top-k/random-k gap  -> a reconstruction VAE keeps task-irrelevant
      variance, so the top PCs should be less action-selective than DINOv2's.

Part A: single / Δf / pair / pair_shuf (GPU ridge).
Part B: top-k / random-k / bottom-k null (GPU SVD + refit), clean condition.
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

KS = (8, 16, 32, 64, 128)
N_SEED = 5


def part_a(feats, index, dev, cond="clean"):
    rows = []
    for stride in data.STRIDES:
        itr, jtr, yc_tr, _ = data.build_pairs(index, stride, "train")
        iva, jva, yc_va, _ = data.build_pairs(index, stride, "val")
        rng = np.random.RandomState(stride)
        ptr, pva = rng.permutation(len(jtr)), rng.permutation(len(jva))

        def build(kind):
            if kind == "single":  return feats[itr], feats[iva]
            if kind == "diff":    return feats[jtr] - feats[itr], feats[jva] - feats[iva]
            if kind == "pair":    return (np.concatenate([feats[itr], feats[jtr]], 1),
                                          np.concatenate([feats[iva], feats[jva]], 1))
            if kind == "pair_shuf": return (np.concatenate([feats[itr], feats[jtr][ptr]], 1),
                                            np.concatenate([feats[iva], feats[jva][pva]], 1))

        for kind in ("single", "diff", "pair", "pair_shuf"):
            if dev == "cuda":
                torch.cuda.empty_cache()
            Ftr, Fva = build(kind)
            r, _ = probes_torch.ridge_probe_t(Ftr, yc_tr, Fva, yc_va, device=dev)
            rows.append(dict(space="wan_vae", condition=cond, stride=stride, part="A",
                             feature=kind, dim=Ftr.shape[1], r2_xyz=r["r2_xyz"],
                             r2_rot=r["r2_rot"], r2_cont6=r["r2_cont6"], alpha=r["alpha"]))
            print(f"[wan:A] {cond:14s} s={stride} {kind:9s} d={Ftr.shape[1]:6d}  "
                  f"xyz={r['r2_xyz']:.3f} rot={r['r2_rot']:.3f} cont6={r['r2_cont6']:.3f}",
                  flush=True)
            del Ftr, Fva
    return rows


def part_b(feats, index, dev, cond="clean"):
    rows = []
    D = feats.shape[1]
    for stride in data.STRIDES:
        itr, jtr, yc_tr, _ = data.build_pairs(index, stride, "train")
        iva, jva, yc_va, _ = data.build_pairs(index, stride, "val")
        Dtr = (feats[jtr] - feats[itr]).astype(np.float32)
        Dva = (feats[jva] - feats[iva]).astype(np.float32)
        full, _ = probes.ridge_probe(Dtr, yc_tr, Dva, yc_va)
        Gtr, Gva = torch.from_numpy(Dtr).to(dev), torch.from_numpy(Dva).to(dev)
        _, _, Vh = torch.linalg.svd(Gtr - Gtr.mean(0, keepdim=True), full_matrices=False)

        def pr(Q):
            res, _ = probes.ridge_probe((Gtr @ Q).cpu().numpy(), yc_tr,
                                        (Gva @ Q).cpu().numpy(), yc_va)
            return res["r2_xyz"]

        for k in KS:
            top, bot = pr(Vh[:k].T), pr(Vh[-k:].T)
            rand = np.array([pr(torch.linalg.qr(torch.randn(
                D, k, generator=torch.Generator(dev).manual_seed(s), device=dev))[0])
                for s in range(N_SEED)])
            rows.append(dict(space="wan_vae", condition=cond, stride=stride, part="B",
                             k=k, D=D, r2_xyz_full=full["r2_xyz"], r2_xyz_top=top,
                             r2_xyz_rand_mean=float(rand.mean()),
                             r2_xyz_rand_std=float(rand.std()), r2_xyz_bottom=bot,
                             top_over_rand=top / rand.mean() if rand.mean() > 1e-9 else float("nan")))
            print(f"[wan:B] {cond:14s} s={stride} k={k:3d}  top={top:.3f} "
                  f"rand={rand.mean():.3f}±{rand.std():.3f} bottom={bot:.3f}  "
                  f"top/rand={rows[-1]['top_over_rand']:.2f}x", flush=True)
        del Gtr, Gva, Vh, Dtr, Dva
        if dev == "cuda":
            torch.cuda.empty_cache()
    return rows


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        free, tot = torch.cuda.mem_get_info()
        print(f"[wan] GPU {torch.cuda.get_device_name(0)} free {free/2**30:.1f} GiB", flush=True)
    index = data.build_index()
    A, B = [], []
    t0 = time.time()
    for cond in CONDITIONS:
        path = os.path.join(data.CACHE_DIR, f"wfeats_{cond}.npy")
        if not os.path.exists(path):
            print(f"[wan] {cond}: {path} missing -- run features_wan first"); continue
        feats = np.load(path)
        print(f"[wan] {cond}: {feats.shape}", flush=True)
        A += part_a(feats, index, dev, cond)
        B += part_b(feats, index, dev, cond)
        del feats
    pd.DataFrame(A).to_csv(os.path.join(data.RESULTS_DIR, "exp_wan_probes.csv"), index=False)
    pd.DataFrame(B).to_csv(os.path.join(data.RESULTS_DIR, "exp_wan_null.csv"), index=False)
    print(f"[wan] done [{time.time()-t0:.0f}s] -> results/exp_wan_probes.csv, exp_wan_null.csv")


if __name__ == "__main__":
    main()
