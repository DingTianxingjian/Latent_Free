"""Patch-token features from the same frozen DINOv2 ViT-B/14.

Exp 1 used the CLS token, a single 768-d global summary.  Exp 3 then found that
UniVLA's stage-1 LAM -- which reads the full 16x16 grid of patch tokens from two
frames -- decodes actions far better than CLS differences do.  That makes the
CLS bottleneck, not the frame difference, the prime suspect for the low Exp-1
numbers.  This module extracts the patch tokens so the comparison can be made in
a matched way.

Two poolings are cached, both derived from the SAME forward pass:

  mean : mean over all 256 patch tokens -> 768-d.  Dimension-matched to CLS, so
         `pmean` vs `cls` isolates "which pooling" with probe capacity held
         fixed.  Destroys spatial layout.
  g4   : 4x4 average pool of the 16x16 token grid -> 16*768 = 12288-d.  Keeps a
         coarse "where is the arm" signal, which is what a translation readout
         needs, and is much closer to what the LAM actually sees.

Nothing else changes: same encoder weights, same 224x224 resize, same ImageNet
normalisation, same distractor synthesis (imported from features.py), same
frame ordering.  The extractor asserts that the CLS token it recomputes matches
the one cached by features.py, so any drift in the pipeline fails loudly.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import data
import features

POOLS = ("mean", "g4")
GRID = 16                       # 224 / 14
G4 = 4                          # 4x4 pooled grid
DIMS = {"mean": 768, "g4": G4 * G4 * 768}


def pool_path(condition, pool):
    return os.path.join(data.CACHE_DIR, f"pfeats_{condition}_{pool}.npy")


def pool_tokens(tok):
    """(B,256,768) patch tokens -> dict of pooled features."""
    B, N, D = tok.shape
    assert N == GRID * GRID, tok.shape
    grid = tok.reshape(B, GRID, GRID, D).permute(0, 3, 1, 2)        # (B,D,16,16)
    g4 = F.avg_pool2d(grid, GRID // G4)                             # (B,D,4,4)
    return {
        "mean": tok.mean(dim=1),                                    # (B,768)
        # (B,4,4,D) -> flat, so consecutive entries share a spatial cell
        "g4": g4.permute(0, 2, 3, 1).reshape(B, -1),
    }


def extract_condition(condition, pools=POOLS, limit_episodes=None, batch=64,
                      device="cuda", check_cls=True):
    index = data.build_index()
    eps = data.list_episodes()
    if limit_episodes:
        eps = eps[:limit_episodes]

    model = features.load_encoder(device)
    video = features.load_video(device)
    bmask, b = features.border_mask(device)
    mean = torch.tensor(features.IMNET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(features.IMNET_STD, device=device).view(1, 3, 1, 1)

    n_total = int(index["lengths"][:len(eps)].sum())
    out = {p: np.zeros((n_total, DIMS[p]), dtype=np.float32) for p in pools}
    print(f"[{condition}] {len(eps)} episodes, {n_total} frames -> "
          + ", ".join(f"{p}:{DIMS[p]}d" for p in pools), flush=True)

    # consistency check against the CLS features exp1/exp2 were built on
    ref = None
    if check_cls:
        rp = os.path.join(data.CACHE_DIR, f"feats_{condition}.npy")
        if os.path.exists(rp):
            ref = np.load(rp, mmap_mode="r")
            assert ref.shape[0] == n_total or limit_episodes, ref.shape
    cls_err = []

    row = 0
    for n, ep in enumerate(eps):
        raw = data.load_episode_frames(ep)
        x = torch.from_numpy(raw).to(device).permute(0, 3, 1, 2).float().div_(255.0)
        x = F.interpolate(x, size=(features.IMG_SIZE, features.IMG_SIZE),
                          mode="bilinear", align_corners=False)
        x = features.apply_distractor(x, condition, ep["ep_id"], video, bmask, b)
        x = (x - mean) / std

        acc = {p: [] for p in pools}
        cls = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                             enabled=(device == "cuda")):
            for k in range(0, x.shape[0], batch):
                f = model.forward_features(x[k:k + batch])
                pooled = pool_tokens(f["x_norm_patchtokens"].float())
                for p in pools:
                    acc[p].append(pooled[p].cpu().numpy())
                if ref is not None:
                    cls.append(f["x_norm_clstoken"].float().cpu().numpy())

        T = raw.shape[0]
        for p in pools:
            v = np.concatenate(acc[p])
            assert v.shape == (T, DIMS[p]), (p, v.shape)
            out[p][row:row + T] = v
        if ref is not None:
            c = np.concatenate(cls)
            r = np.asarray(ref[row:row + T])
            cos = ((c * r).sum(1) /
                   (np.linalg.norm(c, axis=1) * np.linalg.norm(r, axis=1) + 1e-8))
            cls_err.append((float(np.abs(c - r).max() / (np.abs(r).max() + 1e-8)),
                            float(cos.min())))
        row += T
        if (n + 1) % 50 == 0 or n == len(eps) - 1:
            print(f"  [{condition}] {n + 1}/{len(eps)} episodes, {row} frames",
                  flush=True)

    assert row == n_total
    if cls_err:
        e = max(x for x, _ in cls_err)
        cos = min(c for _, c in cls_err)
        print(f"[{condition}] CLS reproducibility vs cached feats_{condition}.npy: "
              f"max rel err {e:.2e}, min cosine {cos:.6f}")
        # fp16 autocast is not bitwise reproducible across batch sizes (this runs
        # at 64, features.py ran at 128), so an exact match is not expected.
        # Cosine is the meaningful test: it stays ~1 under fp16 jitter and under
        # any pure rescaling, and collapses if the frames or the ordering differ.
        assert cos > 0.99, (
            f"patch extractor diverged from the cached CLS run "
            f"(min cosine {cos:.4f}, max rel err {e:.2e}) -- the frames, their "
            f"order, or the distractor synthesis do not match exp 1")
        if e > 5e-2:
            print(f"[{condition}] WARNING: same direction but rel err {e:.2e} -- "
                  f"check for a scale difference in the encoder head")

    for p in pools:
        dst = pool_path(condition, p)
        # write-then-rename: two jobs may extract concurrently (the same run is
        # queued on both `ai` and `gpu`), and a half-written .npy that another
        # process then loads would be silent corruption rather than an error.
        tmp = f"{dst}.tmp{os.getpid()}"
        np.save(tmp, out[p])
        os.replace(tmp + ".npy" if not tmp.endswith(".npy") else tmp, dst)
        print(f"[{condition}] saved {dst} {out[p].shape} "
              f"norm mean {np.linalg.norm(out[p], axis=1).mean():.2f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["clean"])
    ap.add_argument("--pools", nargs="+", default=list(POOLS))
    ap.add_argument("--limit-episodes", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()

    os.makedirs(data.CACHE_DIR, exist_ok=True)
    data.build_index()
    t0 = time.time()
    for c in a.conditions:
        todo = [p for p in a.pools if a.redo or not os.path.exists(pool_path(c, p))]
        if not todo:
            print(f"[{c}] all pools cached, skipping")
            continue
        t1 = time.time()
        extract_condition(c, todo, a.limit_episodes, a.batch, a.device)
        print(f"[{c}] took {(time.time() - t1) / 60:.2f} min", flush=True)
    print(f"[features_patch] done in {(time.time() - t0) / 60:.2f} min")
