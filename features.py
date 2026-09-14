"""Frozen-encoder feature extraction + image-level distractor synthesis.

Encoder: DINOv2 ViT-B/14, CLS token (768-d).  Weights are pre-fetched to
cache/torch so the compute node never needs network access.

Distractor conditions (all synthesised at the image level -- the simulator is
never touched):

  clean         : untouched frames.
  patch         : a 64x64 crop of a natural video composited onto the frame
                  border, its centre travelling at constant velocity around the
                  image perimeter with wrap-around, and the video itself playing.
                  Motion is temporally coherent within an episode and completely
                  independent of the robot.
  border        : the outer 20% ring of every frame is replaced by a playing
                  natural video (the "stronger" condition).
  patch_static  : CONTROL. Same geometry as `patch` but the patch neither moves
  border_static   nor plays -- it is frozen for the whole episode.  A frozen
                  occluder contributes ~no temporal variance, so it cancels out
                  of a frame difference.  Comparing patch vs patch_static
                  separates "the distractor moves" from "the distractor hides
                  pixels", which the moving-only conditions confound.

Distractors are composited AFTER the resize to 224x224, i.e. in the resolution
the encoder actually sees (LIBERO frames are natively 128x128, so a 64x64 patch
applied natively would cover a quarter of the image).
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import data

IMG_SIZE = 224
PATCH_SIZE = 64
BORDER_FRAC = 0.20          # ring thickness as a fraction of the image side
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)

CONDITIONS = ("clean", "patch", "patch_static", "border", "border_static",
              "patch_rand")
# `patch_rand` is ADDITIVE -- every pre-existing condition is byte-identical, so
# earlier results stay reproducible.  Same patch size and same source video as
# `patch`, but the position is drawn independently per frame instead of sweeping
# the perimeter at constant velocity.  It exists so a method fitted against one
# motion law can be evaluated against a DIFFERENT one; evaluating on the same
# distractor a projection was estimated from is circular.

TORCH_HOME = os.path.join(data.ROOT, "cache", "torch")
DINOV2_REPO = os.path.join(TORCH_HOME, "hub", "facebookresearch_dinov2_main")
DINOV2_CKPT = os.path.join(TORCH_HOME, "hub", "checkpoints",
                           "dinov2_vitb14_pretrain.pth")
VIDEO_NPY = os.path.join(data.ROOT, "data", "distractor_video.npy")


# --------------------------------------------------------------------------- #
# distractor synthesis
# --------------------------------------------------------------------------- #
def load_video(device):
    """Natural video clip resized to IMG_SIZE, as (n,3,H,W) float in [0,1]."""
    v = np.load(VIDEO_NPY)                                  # (n,720,1280,3) uint8
    t = torch.from_numpy(v).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                      align_corners=False, antialias=True)
    return t.contiguous()


def border_mask(device):
    """(1,1,H,W) mask that is 1 on the outer BORDER_FRAC ring."""
    b = int(round(BORDER_FRAC * IMG_SIZE))
    m = torch.ones(1, 1, IMG_SIZE, IMG_SIZE, device=device)
    m[:, :, b:IMG_SIZE - b, b:IMG_SIZE - b] = 0.0
    return m, b


def perimeter_topleft(u):
    """Map arc-length u onto the top-left corner of a patch whose centre runs
    around the image perimeter (inset by PATCH_SIZE/2 so it stays in frame)."""
    h = PATCH_SIZE // 2
    lo, hi = h, IMG_SIZE - h
    side = hi - lo
    u = u % (4 * side)
    if u < side:
        cx, cy = lo + u, lo
    elif u < 2 * side:
        cx, cy = hi, lo + (u - side)
    elif u < 3 * side:
        cx, cy = hi - (u - 2 * side), hi
    else:
        cx, cy = lo, hi - (u - 3 * side)
    x = int(round(cx)) - h
    y = int(round(cy)) - h
    return (max(0, min(IMG_SIZE - PATCH_SIZE, x)),
            max(0, min(IMG_SIZE - PATCH_SIZE, y)))


def episode_distractor_params(ep_id, n_video):
    """Deterministic per-episode distractor parameters (robot-independent)."""
    rng = np.random.RandomState(10_000 + int(ep_id))
    return dict(
        v0=int(rng.randint(n_video)),                       # video start frame
        u0=float(rng.uniform(0, 4 * (IMG_SIZE - PATCH_SIZE))),
        speed=float(rng.uniform(4.0, 8.0)) * (1 if rng.rand() < 0.5 else -1),
        crop_x=int(rng.randint(0, IMG_SIZE - PATCH_SIZE)),
        crop_y=int(rng.randint(0, IMG_SIZE - PATCH_SIZE)),
    )


# --------------------------------------------------------------------------- #
# camera-induced motion  (UniVLA names this failure mode explicitly:
# "the emergence of new objects, movements of external agents, or CAMERA-INDUCED
#  MOTION ARTIFACTS").  Every distractor in this repo until now pastes CONTENT --
# a 63px patch or a border strip -- which is a LOCAL occluder.  Camera motion is
# GLOBAL: it moves every pixel, including the agent, without the agent causing it.
# Bridge and LIBERO both have a fixed camera per episode, so the repo had no way to
# exhibit this mode at all; a null measured with the patch says nothing about it.
# Parameters are deterministic in ep_id, exactly like episode_distractor_params.
# --------------------------------------------------------------------------- #
def episode_camera_params(ep_id):
    rs = np.random.RandomState(30_000 + int(ep_id))
    return dict(ax=rs.uniform(0.05, 0.12), ay=rs.uniform(0.05, 0.12),
                fx=rs.uniform(0.03, 0.09), fy=rs.uniform(0.03, 0.09),
                px=rs.uniform(0, 2 * np.pi), py=rs.uniform(0, 2 * np.pi),
                rot=rs.uniform(0.01, 0.04), fr=rs.uniform(0.02, 0.06))


def camera_offset(p, t):
    """Normalised (dx, dy, angle) of the virtual camera at frame index t."""
    t = np.asarray(t, np.float64)
    return (p["ax"] * np.sin(p["fx"] * t + p["px"]),
            p["ay"] * np.sin(p["fy"] * t + p["py"]),
            p["rot"] * np.sin(p["fr"] * t))


def apply_distractor(frames, condition, ep_id, video, bmask, b):
    """frames: (T,3,H,W) float in [0,1] on GPU.  Returns a corrupted copy."""
    if condition == "clean":
        return frames
    T = frames.shape[0]
    p = episode_distractor_params(ep_id, video.shape[0])
    out = frames.clone()
    moving = not condition.endswith("_static")

    if condition.startswith("camera"):
        p = episode_camera_params(ep_id)
        ts = np.arange(T) if moving else np.zeros(T)
        dx, dy, an = camera_offset(p, ts)
        c, s_ = np.cos(an), np.sin(an)
        th = np.zeros((T, 2, 3), np.float32)
        th[:, 0, 0], th[:, 0, 1], th[:, 0, 2] = c, -s_, dx
        th[:, 1, 0], th[:, 1, 1], th[:, 1, 2] = s_, c, dy
        g = torch.nn.functional.affine_grid(
            torch.from_numpy(th).to(frames.device), list(out.shape), align_corners=False)
        return torch.nn.functional.grid_sample(
            out, g, mode="bilinear", padding_mode="border", align_corners=False)

    if condition.startswith("border"):
        idx = (p["v0"] + (np.arange(T) if moving else np.zeros(T, int))) % video.shape[0]
        vid = video[torch.from_numpy(idx).to(frames.device)]
        out = out * (1 - bmask) + vid * bmask
        return out

    # patch conditions
    if condition == "patch_rand":
        # independent uniform position per frame: unpredictable from history, so a
        # method that merely learned `patch`'s constant-velocity sweep fails here.
        rs = np.random.RandomState(20_000 + int(ep_id))
        us = rs.uniform(0, 4 * (IMG_SIZE - PATCH_SIZE), size=T)
    for t in range(T):
        vi = (p["v0"] + (t if moving else 0)) % video.shape[0]
        src = video[vi, :, p["crop_y"]:p["crop_y"] + PATCH_SIZE,
                    p["crop_x"]:p["crop_x"] + PATCH_SIZE]
        if condition == "patch_rand":
            u = float(us[t])
        else:
            u = p["u0"] + (p["speed"] * t if moving else 0.0)
        x, y = perimeter_topleft(u)
        out[t, :, y:y + PATCH_SIZE, x:x + PATCH_SIZE] = src
    return out


# --------------------------------------------------------------------------- #
# encoder
# --------------------------------------------------------------------------- #
def load_encoder(device):
    os.environ.setdefault("TORCH_HOME", TORCH_HOME)
    if not os.path.isdir(DINOV2_REPO):
        raise FileNotFoundError(f"dinov2 hub repo missing at {DINOV2_REPO}")
    model = torch.hub.load(DINOV2_REPO, "dinov2_vitb14", source="local",
                           pretrained=False)
    sd = torch.load(DINOV2_CKPT, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not [k for k in missing if "mask_token" not in k], f"missing keys {missing}"
    print(f"[encoder] DINOv2 ViT-B/14 loaded from {DINOV2_CKPT} "
          f"(missing={list(missing)}, unexpected={list(unexpected)})")
    return model.to(device).eval()


def extract_condition(condition, limit_episodes=None, batch=128, device="cuda"):
    """Extract CLS features for every frame of the suite under `condition`."""
    index = data.build_index()
    eps = data.list_episodes()
    if limit_episodes:
        eps = eps[:limit_episodes]

    model = load_encoder(device)
    video = load_video(device)
    bmask, b = border_mask(device)
    mean = torch.tensor(IMNET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMNET_STD, device=device).view(1, 3, 1, 1)

    n_total = int(index["lengths"][:len(eps)].sum())
    feats = np.zeros((n_total, 768), dtype=np.float32)
    print(f"[{condition}] {len(eps)} episodes, {n_total} frames "
          f"(border ring = {b}px/side, {1 - ((IMG_SIZE - 2 * b) / IMG_SIZE) ** 2:.1%} "
          f"of pixels replaced)")

    row = 0
    for n, ep in enumerate(eps):
        raw = data.load_episode_frames(ep)                  # (T,128,128,3) uint8
        x = torch.from_numpy(raw).to(device).permute(0, 3, 1, 2).float().div_(255.0)
        x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                          align_corners=False)
        x = apply_distractor(x, condition, ep["ep_id"], video, bmask, b)
        x = (x - mean) / std

        outs = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                             enabled=(device == "cuda")):
            for k in range(0, x.shape[0], batch):
                outs.append(model(x[k:k + batch]).float())
        f = torch.cat(outs).cpu().numpy()
        assert f.shape == (raw.shape[0], 768), f.shape
        feats[row:row + f.shape[0]] = f
        row += f.shape[0]
        if (n + 1) % 50 == 0 or n == len(eps) - 1:
            print(f"  [{condition}] {n + 1}/{len(eps)} episodes, {row} frames",
                  flush=True)

    assert row == n_total
    out = os.path.join(data.CACHE_DIR, f"feats_{condition}.npy")
    np.save(out, feats)
    print(f"[{condition}] saved {out}  {feats.shape}  "
          f"norm mean {np.linalg.norm(feats, axis=1).mean():.2f}")
    return out


def save_example_frames(device="cuda"):
    """Write example composited frames for the slides."""
    os.makedirs(data.FIGS_DIR, exist_ok=True)
    import imageio.v3 as iio
    eps = data.list_episodes()
    video = load_video(device)
    bmask, b = border_mask(device)
    for ep in (eps[0], eps[137]):
        raw = data.load_episode_frames(ep)
        x0 = torch.from_numpy(raw).to(device).permute(0, 3, 1, 2).float().div_(255.)
        x0 = F.interpolate(x0, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                           align_corners=False)
        for cond in CONDITIONS:
            x = apply_distractor(x0, cond, ep["ep_id"], video, bmask, b)
            for t in (0, min(30, x.shape[0] - 1), min(60, x.shape[0] - 1)):
                im = (x[t].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
                iio.imwrite(os.path.join(
                    data.FIGS_DIR,
                    f"example_{cond}_ep{ep['ep_id']}_t{t}.png"), im)
    print(f"[examples] wrote example frames to {data.FIGS_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--limit-episodes", type=int, default=None)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--examples", action="store_true")
    ap.add_argument("--redo", action="store_true",
                    help="re-extract conditions that are already cached")
    ap.add_argument("--budget-min", type=float, default=None,
                    help="stop before starting a condition that will not fit in "
                         "the remaining wall time (for short debug queues)")
    a = ap.parse_args()

    os.makedirs(data.CACHE_DIR, exist_ok=True)
    data.build_index()
    if a.examples:
        save_example_frames(a.device)

    t0, took, pending = time.time(), [], []
    for c in a.conditions:
        out = os.path.join(data.CACHE_DIR, f"feats_{c}.npy")
        if os.path.exists(out) and not a.redo:
            print(f"[{c}] already cached, skipping ({out})")
            continue
        if a.budget_min is not None and took:
            left = a.budget_min * 60 - (time.time() - t0)
            est = max(took) * 1.15
            if est > left:
                print(f"[{c}] DEFERRED: needs ~{est / 60:.1f} min but only "
                      f"{left / 60:.1f} min of budget left; resubmit to continue.")
                pending.append(c)
                continue
        t1 = time.time()
        extract_condition(c, a.limit_episodes, a.batch, a.device)
        took.append(time.time() - t1)
        print(f"[{c}] took {took[-1] / 60:.2f} min", flush=True)

    print(f"[features] done in {(time.time() - t0) / 60:.2f} min; "
          f"still pending: {pending if pending else 'none'}")
