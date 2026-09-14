"""Exp B -- is the temporal-fusion advantage a general property of VIDEO encoders,
or specific to Wan?  And does V-JEPA2 (our most distractor-robust space) get it?

Exp 6 fed V-JEPA2 a DUPLICATED-frame tubelet [o_t,o_t] -- a video model used as an
image model.  This feeds GENUINE temporal clips and asks the Exp-5b question in
V-JEPA2 space.  V-JEPA2's tubelet is 2, so a (4m+1)-frame window is padded to
(4m+2) by duplicating the last frame (all 4m+1 real frames are seen; the span is
[t, t+4m], stated in the output).  Readouts, spatially pooled with g4 (4x4, layout
kept -- never global-mean; T3 showed pooling costs ~half):

  vj_fused_last  g4 of the LAST temporal token        16384   the fused latent (mirror of Wan zlast)
  vj_fused_tmean g4 mean over temporal tokens          16384   fused, temporally averaged
  vj_single      cached [o_t,o_t] g4                   16384   state baseline (Exp 6 path)
  vj_diff        vj_single[j] - vj_single[i]           16384   naive differencing
  vj_pair        concat(vj_single[i], vj_single[j])    32768   independent endpoints
  vj_pair_shuf   concat(vj_single[i], random)          32768   state-only control
  multi_k        concat of k independent vj_single     k*16384 frame-count NULL (Exp A in this space)
  multi_k_pca    multi_k -> top-16384 train PCs        16384   width-matched null

FUTURE-LEAKAGE CAVEAT (reported).  V-JEPA2's attention is bidirectional, so within
the clip every timestep has seen the others.  But our window ENDS at the target
endpoint o_{t+4m}: no post-target frames are ever visible, so the window is already
TAIL-ALIGNED and cannot leak motion beyond the interval being decoded.  What the
bidirectionality does allow is that o_t's tokens have seen o_{t+4m} -- that is the
fusion we are measuring, not leakage past the target.  (Wan's path is causal;
V-JEPA2's is not -- noted in the side-by-side.)

Verdict rule (printed):
  * vj_fused >> vj_diff at matched width AND matched frames (multi_5 does NOT reach
    vj_fused) -> fusion is general to video encoders; combined with V-JEPA2's better
    distractor robustness, the method learns z in V-JEPA2 space.
  * vj_fused ~= vj_diff, OR multi_5 reaches vj_fused -> fusion is Wan-specific / a
    frame-count effect here; flag the fusion-vs-robustness trade-off, don't paper over it.

vj_single = cached vjfeats_clean_g4.npy; only vj_fused needs new encoding.
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import data
import probes_torch
from features_vjepa import (GRID, G4, HID, IMG_SIZE, IMNET_MEAN, IMNET_STD,
                            load_encoder)

MS = (1, 2)
KS = {1: (3, 5), 2: (3, 5, 9)}
SINGLE_DIM = G4 * G4 * HID            # 16384 -- width of vj_single / vj_fused
RAW_CAP = 50000                       # ridge gram is d*d f64; raw only below this


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
    """(b, Wp, 128,128,3) uint8 -> (b, Wp, 3, 256,256) ImageNet-normed float."""
    b, Wp = clips_u8.shape[:2]
    x = torch.from_numpy(clips_u8).to(device).permute(0, 1, 4, 2, 3).float().div_(255.0)
    x = x.reshape(b * Wp, 3, 128, 128)
    x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMNET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMNET_STD, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x.reshape(b, Wp, 3, IMG_SIZE, IMG_SIZE)


@torch.no_grad()
def encode_fused(model, FR, starts, W, device, batch=16):
    """Encode the (4m+1)-frame window at each start as a real clip; pad last frame
    to make the frame count even (tubelet=2).  Return (last, tmean) g4 latents."""
    Wp = W if W % 2 == 0 else W + 1
    n = len(starts)
    last = np.zeros((n, SINGLE_DIM), np.float32)
    tmean = np.zeros((n, SINGLE_DIM), np.float32)
    t0 = time.time()
    for k in range(0, n, batch):
        idx = starts[k:k + batch]
        buf = np.zeros((len(idx), Wp, 128, 128, 3), np.uint8)
        for r, s in enumerate(idx):
            w = FR[s:s + W]                                   # (W,128,128,3)
            buf[r, :W] = w
            if Wp > W:
                buf[r, W:] = w[-1]                            # duplicate last frame
        clip = _preprocess(buf, device)                      # (b,Wp,3,256,256)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            tok = model(pixel_values_videos=clip).last_hidden_state   # (b, T'*256, 1024)
        tok = tok.float()
        Tt = tok.shape[1] // (GRID * GRID)
        assert Tt == Wp // 2, (Tt, Wp)
        g = tok.reshape(len(idx), Tt, GRID, GRID, HID).permute(0, 1, 4, 2, 3)  # (b,T',D,16,16)
        g = g.reshape(len(idx) * Tt, HID, GRID, GRID)
        g4 = F.avg_pool2d(g, GRID // G4)                                      # (b*T', D, 4, 4)
        # flatten as (h, w, D) to match features_vjepa.pool_tokens g4 layout
        g4 = g4.reshape(len(idx), Tt, HID, G4, G4).permute(0, 1, 3, 4, 2)     # (b,T',4,4,D)
        g4 = g4.reshape(len(idx), Tt, -1)                                     # (b,T',16384)
        last[k:k + len(idx)] = g4[:, -1].cpu().numpy()
        tmean[k:k + len(idx)] = g4.mean(1).cpu().numpy()
        if (k // batch) % 200 == 0:
            print(f"    fused {k+len(idx)}/{n} [{time.time()-t0:.0f}s]", flush=True)
    return last, tmean


def smoke(model, FR, S, device):
    """fused[o_t,o_t] must reproduce the cached vj_single.  W=1 -> pad-last builds
    [o_t,o_t] (the exact Exp-6 duplicated-frame tubelet); W=2 would feed two
    *different* frames and legitimately not match."""
    s = np.array([100])
    last, _ = encode_fused(model, FR, s, 1, device, batch=1)   # W=1 -> Wp=2 pad-last = [o_t,o_t]
    d = np.abs(last[0] - S[100]).max()
    rel = d / (np.abs(S[100]).mean() + 1e-6)
    print(f"[B] smoke fused[o_t,o_t] vs cached single: max|d|={d:.3f} rel={rel:.3f}", flush=True)
    assert rel < 0.05, f"fused [o_t,o_t] should match cached vj_single (rel={rel})"


def pca_project(Xtr, Xva, q, device):
    """Top-q train PCs.  Center by TRAIN mean; free train before loading val so
    the wide multi_9 case (~147k-d) never holds both matrices on the GPU at once."""
    Xt = torch.as_tensor(np.ascontiguousarray(Xtr), dtype=torch.float32, device=device)
    mu = Xt.mean(0, keepdim=True)
    Xt -= mu
    _, _, V = torch.pca_lowrank(Xt, q=q, center=False, niter=4)
    Zt = (Xt @ V).cpu().numpy()
    del Xt
    if device == "cuda":
        torch.cuda.empty_cache()
    Xv = torch.as_tensor(np.ascontiguousarray(Xva), dtype=torch.float32, device=device)
    Xv -= mu
    Zv = (Xv @ V).cpu().numpy()
    del Xv, V
    if device == "cuda":
        torch.cuda.empty_cache()
    return Zt, Zv


def probe(Ftr, ytr, Fva, yva, dev, feat, m, note, rows):
    r, _ = probes_torch.ridge_probe_t(Ftr.astype(np.float32), ytr,
                                      Fva.astype(np.float32), yva, device=dev)
    rows.append(dict(m=m, span=4 * m, feature=feat, dim=Ftr.shape[1],
                     r2_xyz=r["r2_xyz"], r2_rot=r["r2_rot"], r2_cont6=r["r2_cont6"],
                     alpha=r["alpha"], note=note))
    print(f"[B] m={m} {feat:16s} d={Ftr.shape[1]:6d} xyz={r['r2_xyz']:.3f} "
          f"rot={r['r2_rot']:.3f} cont6={r['r2_cont6']:.3f}  ({note})", flush=True)
    return r


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        free, _ = torch.cuda.mem_get_info()
        print(f"[B] GPU {torch.cuda.get_device_name(0)} free {free/2**30:.1f} GiB", flush=True)
    index = data.build_index()
    S = np.load(os.path.join(data.CACHE_DIR, "vjfeats_clean_g4.npy"))     # (N,16384) vj_single
    assert S.shape[1] == SINGLE_DIM, S.shape
    print(f"[B] loading all frames...", flush=True)
    FR = load_all_frames(index)
    model = load_encoder(dev)
    smoke(model, FR, S, dev)

    rows = []
    fused_xyz = {}
    for m in MS:
        W = 4 * m + 1
        stride = 4 * m
        span_note = f"span={stride} frames={W}(+1 pad); tail-aligned; bidir"
        it, jt, yct, _ = data.build_pairs(index, stride, "train")
        iv, jv, ycv, _ = data.build_pairs(index, stride, "val")
        assert (jt == it + stride).all()

        # --- fused (the only new encoding); cache per (m, split) ---
        for tag, starts in (("train", it), ("val", iv)):
            if not os.path.exists(os.path.join(data.CACHE_DIR, f"vjfused_m{m}_{tag}_last.npy")):
                last, tmean = encode_fused(model, FR, starts, W, dev)
                np.save(os.path.join(data.CACHE_DIR, f"vjfused_m{m}_{tag}_last.npy"), last)
                np.save(os.path.join(data.CACHE_DIR, f"vjfused_m{m}_{tag}_tmean.npy"), tmean)
        L = {sp: np.load(os.path.join(data.CACHE_DIR, f"vjfused_m{m}_{sp}_last.npy"))
             for sp in ("train", "val")}
        Tm = {sp: np.load(os.path.join(data.CACHE_DIR, f"vjfused_m{m}_{sp}_tmean.npy"))
              for sp in ("train", "val")}

        rng = np.random.RandomState(stride)
        ptr, pva = rng.permutation(len(jt)), rng.permutation(len(jv))
        offs = {k: np.unique(np.linspace(0, 4 * m, k).round().astype(int)) for k in KS[m]}

        # --- baselines from cached vj_single ---
        probe(S[it], yct, S[iv], ycv, dev, "vj_single", m, span_note, rows)
        probe(S[jt] - S[it], yct, S[jv] - S[iv], ycv, dev, "vj_diff", m, span_note, rows)
        probe(np.concatenate([S[it], S[jt]], 1), yct,
              np.concatenate([S[iv], S[jv]], 1), ycv, dev, "vj_pair", m, span_note, rows)
        probe(np.concatenate([S[it], S[jt][ptr]], 1), yct,
              np.concatenate([S[iv], S[jv][pva]], 1), ycv, dev, "vj_pair_shuf", m, span_note, rows)

        # --- fused ---
        rf = probe(L["train"], yct, L["val"], ycv, dev, "vj_fused_last", m, span_note, rows)
        probe(Tm["train"], yct, Tm["val"], ycv, dev, "vj_fused_tmean", m, span_note, rows)
        fused_xyz[m] = rf["r2_xyz"]

        # --- frame-count null in V-JEPA2 space ---
        for k in KS[m]:
            o = offs[k]
            Xtr = np.concatenate([S[it + off] for off in o], axis=1)
            Xva = np.concatenate([S[iv + off] for off in o], axis=1)
            note = f"frames@{list(map(int, o))}"
            if Xtr.shape[1] <= RAW_CAP:
                probe(Xtr, yct, Xva, ycv, dev, f"multi_{k}", m, note, rows)
            else:
                print(f"[B] multi_{k} raw d={Xtr.shape[1]} > cap; pca-only", flush=True)
            Ztr, Zva = pca_project(Xtr, Xva, SINGLE_DIM, dev)
            probe(Ztr, yct, Zva, ycv, dev, f"multi_{k}_pca{SINGLE_DIM}", m,
                  note + f"|pca{SINGLE_DIM}", rows)
            del Xtr, Xva

    df = pd.DataFrame(rows)
    out = os.path.join(data.RESULTS_DIR, "exp6b_vjepa_temporal.csv")
    df.to_csv(out, index=False)
    print(f"\n[B] wrote {out}\n", flush=True)

    print("=" * 72)
    print("VERDICT -- V-JEPA2 temporal fusion (xyz R2)")
    print("=" * 72)
    for m in MS:
        sub = df[df.m == m]
        def g(f): return float(sub[sub.feature == f].r2_xyz.iloc[0])
        fu, di, si = g("vj_fused_last"), g("vj_diff"), g("vj_single")
        kmax = max(KS[m])
        mk = g(f"multi_{kmax}_pca{SINGLE_DIM}")
        print(f"m={m} span={4*m}: single {si:.3f} | diff {di:.3f} | "
              f"multi_{kmax}_pca {mk:.3f} | fused_last {fu:.3f}")
        print(f"    fused>diff by {fu-di:+.3f}; fused>multi_{kmax}(matched frames+width) by {fu-mk:+.3f}")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
