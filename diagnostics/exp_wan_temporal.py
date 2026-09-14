"""Wan2.2 VAE -- does the video model's NATIVE temporal compression encode action?

Every prior Wan readout (Exp 5) differenced INDEPENDENT single-frame latents
(Delta f); the 4x temporal compression was a deliberate no-op (T=1 -> T'=1).  This
asks the video-model question directly: when the VAE actually FUSES frames through
its causal temporal convs, is the fused latent a *better* motion representation
than naive two-frame differencing / concatenation of independent frame latents?

VAE temporal arithmetic (VideoVAE_.encode): iter_ = 1 + (T-1)//4.  A clip must
span the 4-frame grid to fuse anything: T=2 -> iter_=1 encodes ONLY frame 0 (the
2nd frame is DROPPED).  T=5 -> iter_=2: z0 from o_t (a fresh-cache single-frame
encode), z1 fusing {o_{t+1..t+4}} with causal context of o_t.  So the natural
temporal unit is a (4m+1)-frame window giving m+1 latents; target = net action
over the 4m steps (m=1 == our stride-4 target, m=2 == stride-8).

Features (all decode the SAME y_cont = sum actions[t:t+4m], the net command):
  single    wf[t]                          9408        one-frame state (baseline)
  diff      wf[t+4m]-wf[t]                 9408        naive differencing (Exp 5)
  pair      [wf[t+4g] for g in 0..m]       (m+1)*9408  independent grid snapshots
  pair_shuf [wf[t], wf[random], ...]       (m+1)*9408  state-only control
  video     flatten([z0..zm])             (m+1)*9408  VAE's FUSED window latent
  zlast     z_m alone                      9408        the fused motion latent

Headline: video vs pair at EQUAL width -- does temporal fusion beat independent
snapshots?  zlast vs single -- what fusion adds over one frame.  wf[*] = cached
single-frame latents (cache/wfeats_clean.npy), reused for free.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import data
import probes_torch
from features_wan import IMG_SIZE, LAT_DIM, load_vae

MS = (1, 2)                                   # window = 4m+1 frames -> m+1 latents


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


@torch.no_grad()
def encode_windows(vae, FR, starts, W, device, batch=8):
    """starts: global row ids of the window's first frame; window = FR[s:s+W].
    Returns (n, nlat, LAT_DIM) float32 -- the m+1 temporal latents per window."""
    n = len(starts)
    nlat = 1 + (W - 1) // 4
    out = np.zeros((n, nlat, LAT_DIM), np.float32)
    t0 = time.time()
    for k in range(0, n, batch):
        idx = starts[k:k + batch]
        clips = []
        for s in idx:
            w = FR[s:s + W]                                      # (W,128,128,3) u8
            x = torch.from_numpy(w).to(device).permute(0, 3, 1, 2).float().div_(255.0)
            x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                              align_corners=False)
            x = x.mul_(2.0).sub_(1.0)                            # [-1,1]
            clips.append(x.permute(1, 0, 2, 3).contiguous())    # (3,W,H,W)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            z = vae.encode(clips)                               # (b,48,nlat,14,14)
        z = z.permute(0, 2, 1, 3, 4).reshape(len(clips), nlat, LAT_DIM)
        out[k:k + len(clips)] = z.float().cpu().numpy()
        if (k // batch) % 200 == 0:
            print(f"    encode {k+len(clips)}/{n} [{time.time()-t0:.0f}s]", flush=True)
    return out


def smoke(vae, FR, wf, device):
    """z0 of a 5-frame window must match the standalone single-frame latent."""
    s = np.array([100])
    z = encode_windows(vae, FR, s, 5, device, batch=1)          # (1,2,LAT_DIM)
    d = np.abs(z[0, 0] - wf[100]).max()
    rel = d / (np.abs(wf[100]).mean() + 1e-6)
    print(f"[wanT] smoke z0-vs-single max|d|={d:.3f} rel={rel:.3f}  "
          f"z1!=z0 (fused): {np.abs(z[0,1]-z[0,0]).mean():.3f}", flush=True)
    assert rel < 0.05, f"z0 should equal the standalone single-frame latent (rel={rel})"
    assert np.abs(z[0, 1] - z[0, 0]).mean() > 1e-3, "z1 must differ from z0 (temporal fusion)"


def probe(Ftr, ytr, Fva, yva, dev, name, m, feat, rows):
    r, _ = probes_torch.ridge_probe_t(Ftr, ytr, Fva, yva, device=dev)
    rows.append(dict(m=m, span=4 * m, feature=feat, dim=Ftr.shape[1],
                     r2_xyz=r["r2_xyz"], r2_rot=r["r2_rot"], r2_cont6=r["r2_cont6"],
                     alpha=r["alpha"]))
    print(f"[wanT] m={m} {feat:9s} d={Ftr.shape[1]:6d} xyz={r['r2_xyz']:.3f} "
          f"rot={r['r2_rot']:.3f} cont6={r['r2_cont6']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ms", type=int, nargs="+", default=list(MS))
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        free, _ = torch.cuda.mem_get_info()
        print(f"[wanT] GPU {torch.cuda.get_device_name(0)} free {free/2**30:.1f} GiB", flush=True)

    index = data.build_index()
    wf = np.load(os.path.join(data.CACHE_DIR, "wfeats_clean.npy"))     # (N, 9408)
    assert wf.shape[1] == LAT_DIM, wf.shape
    print(f"[wanT] loading all frames...", flush=True)
    FR = load_all_frames(index)
    vae = load_vae(dev)
    smoke(vae, FR, wf, dev)

    rows = []
    for m in a.ms:
        W = 4 * m + 1
        stride = 4 * m
        packs = {}
        for split in ("train", "val"):
            i, j, yc, _ = data.build_pairs(index, stride, split)      # i=start, j=i+stride
            vid = encode_windows(vae, FR, i, W, dev, a.batch)         # (n, m+1, LAT_DIM)
            snaps = np.stack([wf[i + 4 * g] for g in range(m + 1)], axis=1)  # (n,m+1,LAT_DIM)
            rng = np.random.RandomState(stride)
            perm = rng.permutation(len(i))
            packs[split] = dict(i=i, j=j, yc=yc, vid=vid, snaps=snaps, perm=perm)

        tr, va = packs["train"], packs["val"]

        def feat(p, kind):
            n = len(p["i"])
            if kind == "single":  return wf[p["i"]]
            if kind == "diff":    return wf[p["j"]] - wf[p["i"]]
            if kind == "pair":    return p["snaps"].reshape(n, -1)
            if kind == "pair_shuf":
                s = p["snaps"].copy(); s[:, 1:] = s[p["perm"]][:, 1:]  # keep o_t, shuffle rest
                return s.reshape(n, -1)
            if kind == "video":   return p["vid"].reshape(n, -1)
            if kind == "zlast":   return p["vid"][:, m]                # the fused chunk latent
            if kind == "z0":      return p["vid"][:, 0]

        for kind in ("single", "diff", "pair", "pair_shuf", "video", "zlast", "z0"):
            probe(feat(tr, kind).astype(np.float32), tr["yc"],
                  feat(va, kind).astype(np.float32), va["yc"], dev, "wan", m, kind, rows)

    pd.DataFrame(rows).to_csv(os.path.join(data.RESULTS_DIR, "exp_wan_temporal.csv"),
                              index=False)
    print("[wanT] wrote results/exp_wan_temporal.csv", flush=True)


if __name__ == "__main__":
    main()
