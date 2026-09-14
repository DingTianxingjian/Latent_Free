"""V-JEPA2 features -- a *prediction*-target space (the third encoder objective).

DINOv2: self-distillation.  Wan VAE: pixel reconstruction.  V-JEPA2: masked
*prediction* in latent space, trained on video.  Item-3 pred 4 asks whether a
prediction objective, unlike DINOv2, makes spatial pooling *less* harmful (its
per-token features may already carry motion/location redundantly).

V-JEPA2 tokenizes with a temporal tubelet (size 2).  To stay byte-comparable to
the per-frame DINOv2/Wan protocol we feed each frame as a 2-frame clip [o_t,o_t]
-> 1 temporal x 16x16 spatial = 256 tokens x 1024-d.  RoPE makes the short clip
legal.  Two poolings, both from the same forward pass:

  pmean : mean over the 256 tokens -> 1024-d       (destroys spatial layout)
  g4    : 4x4 avg-pool of the 16x16 grid -> 16*1024 = 16384-d  (keeps layout)

Frame -> resize 256 -> ImageNet norm -> VJEPA2Model -> tokens.  Distractors are
applied at 256 in [0,1] (features.apply_distractor), exactly as the other spaces.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import data
import features

VJEPA_DIR = os.path.join(data.ROOT, "cache", "vjepa2", "vitl")
IMG_SIZE = 256
GRID = 16                       # 256 / 16
G4 = 4
HID = 1024
DIMS = {"pmean": HID, "g4": G4 * G4 * HID}
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


def load_encoder(device):
    from transformers import VJEPA2Model
    model = VJEPA2Model.from_pretrained(VJEPA_DIR, torch_dtype=torch.float32).to(device).eval()
    print(f"[vjepa] loaded {VJEPA_DIR}", flush=True)
    return model


def pool_tokens(tok):
    """(B,256,1024) spatial tokens -> pooled dict."""
    B, N, D = tok.shape
    assert N == GRID * GRID, tok.shape
    grid = tok.reshape(B, GRID, GRID, D).permute(0, 3, 1, 2)        # (B,D,16,16)
    g4 = F.avg_pool2d(grid, GRID // G4)                             # (B,D,4,4)
    return {"pmean": tok.mean(1), "g4": g4.permute(0, 2, 3, 1).reshape(B, -1)}


@torch.no_grad()
def encode_frames(model, frames_u8, device, batch=16,
                  condition="clean", ep_id=0, distractor=None):
    x = torch.from_numpy(frames_u8).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    if condition != "clean":
        video, bmask, b = distractor
        x = features.apply_distractor(x, condition, ep_id, video, bmask, b)
    mean = torch.tensor(IMNET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMNET_STD, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    acc = {p: [] for p in DIMS}
    for k in range(0, x.shape[0], batch):
        chunk = x[k:k + batch]                                      # (b,3,H,W)
        clip = torch.stack([chunk, chunk], dim=1)                  # (b,2,3,H,W) tubelet
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            tok = model(pixel_values_videos=clip).last_hidden_state  # (b,256,1024)
        pooled = pool_tokens(tok.float())
        for p in DIMS:
            acc[p].append(pooled[p].cpu().numpy())
    return {p: np.concatenate(acc[p]) for p in DIMS}


def smoke(model, device):
    fr = data.load_episode_frames(data.list_episodes()[0])[:4]
    z = encode_frames(model, fr, device)
    for p in DIMS:
        assert z[p].shape == (4, DIMS[p]), (p, z[p].shape)
        assert np.isfinite(z[p]).all()
    print(f"[vjepa] smoke OK: pmean {z['pmean'].shape} g4 {z['g4'].shape} "
          f"g4 mean {z['g4'].mean():.3f} std {z['g4'].std():.3f}", flush=True)


def path(cond, pool, lim):
    return os.path.join(data.CACHE_DIR, f"vjfeats_{cond}_{pool}"
                        + ("" if lim is None else f"_lim{lim}") + ".npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=["clean"])
    ap.add_argument("--limit-episodes", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    data.build_index()
    model = load_encoder(a.device)
    smoke(model, a.device)

    eps = data.list_episodes()
    if a.limit_episodes:
        eps = eps[:a.limit_episodes]
    n_total = int(data.build_index()["lengths"][:len(eps)].sum())
    need = any(c != "clean" for c in a.conditions)
    # features.load_video / border_mask / patch placement use features.IMG_SIZE;
    # point it at V-JEPA2's 256 so the mask, video, and border ring are the right size.
    features.IMG_SIZE = IMG_SIZE
    distractor = (features.load_video(a.device), *features.border_mask(a.device)) if need else None

    for cond in a.conditions:
        todo = [p for p in DIMS if a.redo or not os.path.exists(path(cond, p, a.limit_episodes))]
        if not todo:
            print(f"[vjepa] {cond}: cached, skip"); continue
        out = {p: np.zeros((n_total, DIMS[p]), np.float32) for p in DIMS}
        row = 0
        t0 = time.time()
        for n, ep in enumerate(eps):
            fr = data.load_episode_frames(ep)
            z = encode_frames(model, fr, a.device, a.batch, cond, ep["ep_id"], distractor)
            for p in DIMS:
                out[p][row:row + len(fr)] = z[p]
            row += len(fr)
            if (n + 1) % 25 == 0 or n == len(eps) - 1:
                print(f"  [vjepa:{cond}] {n+1}/{len(eps)} eps, {row} frames [{time.time()-t0:.0f}s]",
                      flush=True)
        assert row == n_total
        for p in DIMS:
            np.save(path(cond, p, a.limit_episodes), out[p])
            print(f"[vjepa] saved {path(cond,p,a.limit_episodes)} {out[p].shape}", flush=True)


if __name__ == "__main__":
    main()
