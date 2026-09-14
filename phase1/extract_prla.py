"""PRLA feature extraction: V-JEPA2 fused latents for the HISTORY and FUTURE
windows, encoded as SEPARATE clips (the non-negotiable causality constraint —
V-JEPA2 is bidirectional, so a single long clip would let history see future).

Readout: g4 (4x4 layout kept) pooled over spatial cells per temporal token, then
mean over temporal tokens -> a 16384-d fused latent (the vj_fused_tmean readout,
the strongest in Exp 6b).  Odd frame counts are padded by duplicating the last
frame to satisfy tubelet size 2; history [t-4..t] is 5 frames (->6), future
[t+1..t+4] is 4 frames (already even).

Caches per split, aligned to pdata.prla_windows() order:
  cache/prla_hist_{split}.npy, cache/prla_fut_{split}.npy   (n, 16384)
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data                                    # noqa: E402
import pdata                                    # noqa: E402
from features_vjepa import (GRID, G4, HID, IMG_SIZE, IMNET_MEAN, IMNET_STD,  # noqa: E402
                            load_encoder)

FUSED_DIM = G4 * G4 * HID                       # 16384


def load_all_frames(index):
    n = int(index["lengths"].sum())
    FR = np.zeros((n, 128, 128, 3), np.uint8)
    row = 0
    for ep in data.list_episodes():
        fr = data.load_episode_frames(ep)
        FR[row:row + len(fr)] = fr
        row += len(fr)
    assert row == n, (row, n)
    return FR


def _preprocess(clips_u8, device):
    b, Wp, H, W = clips_u8.shape[0], clips_u8.shape[1], clips_u8.shape[2], clips_u8.shape[3]
    x = torch.from_numpy(clips_u8).to(device).permute(0, 1, 4, 2, 3).float().div_(255.0)
    x = x.reshape(b * Wp, 3, H, W)
    x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMNET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMNET_STD, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.reshape(b, Wp, 3, IMG_SIZE, IMG_SIZE)


@torch.no_grad()
def encode_clips(model, FR, frame_ids, device, batch=32):
    """frame_ids (n, W) global ids -> (n, 16384) tmean-g4 fused latents."""
    n, W = frame_ids.shape
    Wp = W if W % 2 == 0 else W + 1
    out = np.zeros((n, FUSED_DIM), np.float32)
    t0 = time.time()
    for k in range(0, n, batch):
        ids = frame_ids[k:k + batch]                         # (b, W)
        b = len(ids)
        buf = np.zeros((b, Wp) + FR.shape[1:], np.uint8)     # (b,Wp,H,Wpx,3)
        buf[:, :W] = FR[ids]                                 # (b, W, H, Wpx, 3)
        if Wp > W:
            buf[:, W:] = FR[ids[:, -1]][:, None]             # duplicate last frame
        clip = _preprocess(buf, device)                      # (b, Wp, 3, 256,256)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            tok = model(pixel_values_videos=clip).last_hidden_state
        tok = tok.float()
        Tt = tok.shape[1] // (GRID * GRID)
        assert Tt == Wp // 2, (Tt, Wp)
        g = tok.reshape(b, Tt, GRID, GRID, HID).permute(0, 1, 4, 2, 3)   # (b,T',D,16,16)
        g = g.reshape(b * Tt, HID, GRID, GRID)
        g4 = F.avg_pool2d(g, GRID // G4)                      # (b*T',D,4,4)
        g4 = g4.reshape(b, Tt, HID, G4, G4).permute(0, 1, 3, 4, 2)       # (b,T',4,4,D)
        g4 = g4.reshape(b, Tt, -1).mean(1)                   # (b, 16384) tmean
        out[k:k + b] = g4.cpu().numpy()
        if (k // batch) % 200 == 0:
            print(f"    {frame_ids.shape} {k+b}/{n} [{time.time()-t0:.0f}s]", flush=True)
    return out


def smoke(model, FR, device):
    """A 1-frame 'clip' padded to [o_t,o_t] must match a duplicated-frame encode."""
    ids = np.array([[100]])
    a = encode_clips(model, FR, ids, device, batch=1)
    b = encode_clips(model, FR, np.array([[100, 100]]), device, batch=1)
    rel = np.abs(a - b).max() / (np.abs(b).mean() + 1e-6)
    print(f"[prla-x] smoke pad-vs-dup rel={rel:.4f}", flush=True)
    assert rel < 1e-4, rel


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["libero", "bridge"], default="libero")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        free, _ = torch.cuda.mem_get_info()
        print(f"[prla-x] GPU {torch.cuda.get_device_name(0)} free {free/2**30:.1f} GiB", flush=True)
    index = pdata.get_index(a.dataset)
    outdir = os.path.join(data.CACHE_DIR, "bridge") if a.dataset == "bridge" else data.CACHE_DIR
    print(f"[prla-x] dataset={a.dataset}; loading frames...", flush=True)
    if a.dataset == "bridge":
        FR = np.load(os.path.join(data.CACHE_DIR, "bridge", "frames.npy"))   # (N,256,256,3)
    else:
        FR = load_all_frames(index)
    model = load_encoder(dev)
    smoke(model, FR, dev)

    for split in ("train", "val"):
        w = pdata.prla_windows(index, split, a.dataset)
        print(f"[prla-x] {split}: {len(w['t'])} positions", flush=True)
        for name, frames in (("hist", w["hist"]), ("fut", w["fut"])):
            dst = os.path.join(outdir, f"prla_{name}_{split}.npy")
            if os.path.exists(dst):
                print(f"[prla-x] {dst} exists, skip", flush=True)
                continue
            arr = encode_clips(model, FR, frames, dev)
            np.save(dst, arr)
            print(f"[prla-x] saved {dst} {arr.shape} norm {np.linalg.norm(arr,axis=1).mean():.1f}",
                  flush=True)


if __name__ == "__main__":
    main()
