"""Wan2.2 VAE encoder features -- a *reconstruction/generation*-target space.

DINOv2 (Exp 1-4) is trained by self-distillation; V-JEPA2 by masked prediction;
Wan's VAE by pixel reconstruction inside a video generator.  Item 3 asks whether
our conclusions are properties of "frame features" or of DINOv2 specifically.  A
reconstruction VAE is the interesting opposite end: it is trained to preserve
*everything* needed to reproduce the pixels, so task-irrelevant variance is kept
by construction -- the theory predicts (pred. 3) a narrower top-k/random-k gap
and (pred. 2) lower per-dim action decodability than DINOv2.

We use ONLY the VAE encoder latent (not any diffusion-internal feature), the
48-channel Wan2.2 VAE (`WanVideoVAE38`, 16x spatial, 4x temporal).  To keep the
protocol byte-identical to the DINOv2 runs we encode each frame **independently**
as a 1-frame clip, so the 4x temporal compression is a no-op (T=1 -> T'=1) and
"stride" keeps its Exp-1 meaning (a difference of two per-frame latents).  A clip
encoding that engages the temporal axis is a separate follow-up.

Frame -> resize 224 -> [-1,1] -> VAE -> latent 48 x 1 x 14 x 14 = 9408-d.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import data
import features

WAN_MOD = ("/anvil/scratch/x-tding2/surgical_world/dreamzero/"
           "groot/vla/model/dreamzero/modules")
WAN_CKPT = os.path.join(data.ROOT, "cache", "wan", "Wan2.2_VAE.pth")
IMG_SIZE = 224
LAT_DIM = 48 * (IMG_SIZE // 16) ** 2          # 48 * 14 * 14 = 9408


def load_vae(device="cuda"):
    sys.path.insert(0, WAN_MOD)
    from wan_video_vae import WanVideoVAE38
    vae = WanVideoVAE38()                      # builds mean/std on cuda
    sd = torch.load(WAN_CKPT, map_location="cpu")
    missing, unexpected = vae.model.load_state_dict(sd, strict=False)
    assert not missing, f"missing VAE weights: {list(missing)[:8]}"
    vae.model = vae.model.to(device).eval().requires_grad_(False)
    print(f"[wan] loaded {WAN_CKPT}  z_dim={vae.z_dim} up={vae.upsampling_factor} "
          f"unexpected={list(unexpected)[:3]}", flush=True)
    return vae


@torch.no_grad()
def encode_frames(vae, frames_u8, device, batch=8,
                  condition="clean", ep_id=0, distractor=None):
    """(T,128,128,3) uint8 -> (T, LAT_DIM) float32 latents, one frame at a time.

    Distractor synthesis is applied at 224 in [0,1] (exactly as features_patch),
    then the frame is mapped to Wan's [-1,1] input range.
    """
    x = torch.from_numpy(frames_u8).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    if condition != "clean":
        video, bmask, b = distractor
        x = features.apply_distractor(x, condition, ep_id, video, bmask, b)
    x = x.mul_(2.0).sub_(1.0)                                    # [0,1] -> [-1,1]
    out = []
    for k in range(0, x.shape[0], batch):
        chunk = x[k:k + batch]                                   # (b,3,H,W)
        vids = [chunk[i].unsqueeze(1) for i in range(chunk.shape[0])]  # each (3,1,H,W)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            z = vae.encode(vids)                                # (b,48,1,14,14)
        out.append(z.reshape(z.shape[0], -1).float().cpu().numpy())
    return np.concatenate(out)


def smoke(vae, device):
    ep = data.list_episodes()[0]
    fr = data.load_episode_frames(ep)[:4]
    z = encode_frames(vae, fr, device)
    assert z.shape == (4, LAT_DIM), z.shape
    assert np.isfinite(z).all(), "non-finite latents"
    print(f"[wan] smoke OK: 4 frames -> {z.shape}, "
          f"mean {z.mean():.3f} std {z.std():.3f} "
          f"|z| range [{np.abs(z).min():.3f},{np.abs(z).max():.1f}]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["clean"])
    ap.add_argument("--limit-episodes", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    os.makedirs(data.CACHE_DIR, exist_ok=True)
    data.build_index()

    vae = load_vae(a.device)
    smoke(vae, a.device)

    eps = data.list_episodes()
    if a.limit_episodes:
        eps = eps[:a.limit_episodes]
    n_total = int(data.build_index()["lengths"][:len(eps)].sum())

    # distractor resources (shared across conditions), loaded once
    need_distract = any(c != "clean" for c in a.conditions)
    distractor = (features.load_video(a.device), *features.border_mask(a.device)) \
        if need_distract else None

    for cond in a.conditions:
        dst = os.path.join(data.CACHE_DIR, f"wfeats_{cond}"
                           + ("" if a.limit_episodes is None else f"_lim{a.limit_episodes}")
                           + ".npy")
        if os.path.exists(dst) and not a.redo:
            print(f"[wan] {dst} exists, skip"); continue
        out = np.zeros((n_total, LAT_DIM), np.float32)
        row = 0
        t0 = time.time()
        for n, ep in enumerate(eps):
            fr = data.load_episode_frames(ep)
            out[row:row + len(fr)] = encode_frames(vae, fr, a.device, a.batch,
                                                   cond, ep["ep_id"], distractor)
            row += len(fr)
            if (n + 1) % 25 == 0 or n == len(eps) - 1:
                print(f"  [wan:{cond}] {n+1}/{len(eps)} eps, {row} frames "
                      f"[{time.time()-t0:.0f}s]", flush=True)
        assert row == n_total, (row, n_total)
        np.save(dst, out)
        print(f"[wan] saved {dst} {out.shape} norm/mean {np.linalg.norm(out,axis=1).mean():.1f}",
              flush=True)


if __name__ == "__main__":
    main()
