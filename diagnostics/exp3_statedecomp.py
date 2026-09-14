"""Item 1 -- state decomposition of the UniVLA LAM's own latent action.

Takeaway 1 showed that most of what a *two-frame encoder* decodes is single-frame
state.  Does the same hold for a real LAM's latent z?  The decoder already sees
o_t, so any capacity z spends re-encoding o_t's state is wasted -- and that waste
is exactly what a residual design (PRLA's passive predictor P_A) would reclaim.

Method: encode a pair through the LAM two ways and probe z -> action.
  pair     z = LAM(o_t, o_{t+s})            -- the real transition
  shuffled z = LAM(o_t, o_other)            -- o_t intact, transition destroyed
The action target is the TRUE action at t in both cases, so a high shuffled
score means z is reading o_t's state, not the transition.

A random second frame is an OOD input to the encoder, so a low shuffled score
could mean "encoder broke" rather than "no state to read".  To separate those we
use three shuffle tiers of increasing OOD-ness and require a consistent trend:
  same_ep  a far frame from the SAME episode   (same scene, wrong dynamics)
  diff_ep  a frame from a DIFFERENT episode    (different scene)
  rand     a uniformly random frame            (anything)

Both pre-VQ (emb) and post-VQ (z_q) are probed.  The real-pair latents are
reused from exp3's cache; only the three shuffle tiers are encoded here.

Verdict: ratio r = R2(shuffled) / R2(pair), on all-6 R2.
  r high  (>~0.5) and stable across tiers  -> z wastes capacity on state; PRLA
                                               has an empirical target.
  r ~ 0                                     -> the VQ bottleneck already squeezed
                                               state out; PRLA's value prop needs
                                               a redesign.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import data
import probes
from exp3 import load_lam, IMG_SIZE

TIERS = ("same_ep", "diff_ep", "rand")


def rebuild_pair_index(index, stride):
    """Replay exp3's pair loop to recover, per cached pair, the first-frame
    global id and its episode -- exp3 stored neither, so we reconstruct in the
    identical order and assert the targets match."""
    eps = data.list_episodes()
    i_g, ep_of, first_local = [], [], []
    for e in range(len(eps)):
        t, ig, jg, yc, yg = data.episode_pairs(index, e, stride)
        if len(t) == 0:
            continue
        i_g.append(ig)
        first_local.append(t)
        ep_of.append(np.full(len(t), e, np.int64))
    return (np.concatenate(i_g), np.concatenate(ep_of),
            np.concatenate(first_local))


def build_shuffles(index, i_g, ep_of, first_local, stride, seed=0):
    """Second-frame global ids for each tier, aligned with the cached pairs."""
    rng = np.random.RandomState(seed)
    starts, lengths = index["starts"], index["lengths"]
    n_total = int(lengths.sum())
    N = len(i_g)

    # tier 3: uniformly random global frame
    rand = rng.randint(0, n_total, size=N).astype(np.int64)

    # tier 2: a frame from a DIFFERENT episode (resample the few collisions)
    diff = rng.randint(0, n_total, size=N).astype(np.int64)
    for _ in range(8):
        se = starts[ep_of]
        te = se + lengths[ep_of]
        bad = (diff >= se) & (diff < te)
        if not bad.any():
            break
        diff[bad] = rng.randint(0, n_total, size=int(bad.sum()))

    # tier 1: a FAR frame in the SAME episode (|l - t| >= T/3 when possible)
    same = np.empty(N, np.int64)
    for e in np.unique(ep_of):
        m = ep_of == e
        se, Te = int(starts[e]), int(lengths[e])
        tl = first_local[m]
        thr = max(1, Te // 3)
        pick = np.empty(len(tl), np.int64)
        for a in range(len(tl)):
            cand = np.where(np.abs(np.arange(Te) - tl[a]) >= thr)[0]
            if len(cand) == 0:
                cand = np.array([x for x in range(Te) if x != tl[a]])
            pick[a] = se + cand[rng.randint(len(cand))]
        same[m] = pick
    return {"same_ep": same, "diff_ep": diff, "rand": rand}


@torch.no_grad()
def encode(model, first_g, second_g, ALL, device, batch=32):
    """LAM latents (z_q, emb) for the pairs (ALL[first], ALL[second])."""
    Zq, Zc = [], []
    for k in range(0, len(first_g), batch):
        fi = first_g[k:k + batch]
        se = second_g[k:k + batch]
        a = torch.from_numpy(ALL[fi]).to(device).permute(0, 3, 1, 2).float().div_(255.)
        b = torch.from_numpy(ALL[se]).to(device).permute(0, 3, 1, 2).float().div_(255.)
        a = F.interpolate(a, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        b = F.interpolate(b, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        vids = torch.stack([a, b], dim=1)                     # (B,2,3,H,W)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            out = model.vq_encode(vids)
        B = vids.shape[0]
        Zq.append(out["z_q"].reshape(B, -1).float().cpu().numpy())
        Zc.append(out["emb"].reshape(B, -1).float().cpu().numpy())
    return np.concatenate(Zq), np.concatenate(Zc)


def probe(Z, Yc, Yg, val):
    res, _ = probes.run_probe_set(Z[~val], Z[val], Yc[~val], Yc[val],
                                  Yg[~val], Yg[val])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu"

    index = data.build_index()
    eps = data.list_episodes()
    starts, lengths = index["starts"], index["lengths"]
    n_total = int(lengths.sum())
    print(f"[sd] loading {n_total} frames into RAM (uint8)...", flush=True)
    ALL = np.zeros((n_total, 128, 128, 3), np.uint8)
    for e, ep in enumerate(eps):
        ALL[int(starts[e]):int(starts[e]) + int(lengths[e])] = data.load_episode_frames(ep)

    model_box = [None]
    rows = []
    for stride in data.STRIDES:
        cache = np.load(os.path.join(data.CACHE_DIR, f"exp3_z_s{stride}.npz"))
        Zq_pair, Zc_pair = cache["Zq"], cache["Zc"]
        Yc, Yg, val = cache["Yc"], cache["Yg"], cache["val"].astype(bool)

        i_g, ep_of, first_local = rebuild_pair_index(index, stride)
        # alignment guarantee: reconstructed targets must equal the cached ones
        assert len(i_g) == len(Yc), (len(i_g), len(Yc))
        yc_re = np.concatenate([data.episode_pairs(index, e, stride)[3]
                                for e in range(len(eps))
                                if len(data.episode_pairs(index, e, stride)[0])])
        assert np.allclose(yc_re, Yc, atol=1e-5), "pair order mismatch vs exp3 cache"
        print(f"[sd] stride {stride}: {len(i_g)} pairs, alignment OK", flush=True)

        shuf = build_shuffles(index, i_g, ep_of, first_local, stride)

        # real pair (reuse cache), then the three shuffle tiers
        latents = {"pair": (Zq_pair, Zc_pair)}
        for tier in TIERS:
            cpath = os.path.join(data.CACHE_DIR, f"sd_z_s{stride}_{tier}.npz")
            if os.path.exists(cpath):
                d = np.load(cpath)
                latents[tier] = (d["Zq"], d["Zc"])
                print(f"[sd]   loaded {cpath}", flush=True)
            else:
                if model_box[0] is None:
                    model_box[0] = load_lam(dev)
                zq, zc = encode(model_box[0], i_g, shuf[tier], ALL, dev, a.batch)
                np.savez(cpath, Zq=zq, Zc=zc)
                latents[tier] = (zq, zc)
                print(f"[sd]   encoded {tier} -> {cpath} {zq.shape}", flush=True)

        base = {}
        for kind in ("pair",) + TIERS:
            Zq, Zc = latents[kind]
            for lname, Z in (("z_q", Zq), ("emb", Zc)):
                r = probe(Z, Yc, Yg, val)
                if kind == "pair":
                    base[lname] = r
                ratio = (r["r2_cont6"] / base[lname]["r2_cont6"]
                         if base[lname]["r2_cont6"] > 1e-6 else float("nan"))
                rows.append(dict(stride=stride, latent=lname, pairing=kind,
                                 r2_xyz=r["r2_xyz"], r2_rot=r["r2_rot"],
                                 r2_cont6=r["r2_cont6"], bal_acc_grip=r["bal_acc_grip"],
                                 ratio_cont6_vs_pair=ratio, alpha=r["alpha"]))
                print(f"[sd] s={stride} {lname:3s} {kind:8s}  "
                      f"xyz={r['r2_xyz']:.3f} rot={r['r2_rot']:.3f} "
                      f"cont6={r['r2_cont6']:.3f} grip={r['bal_acc_grip']:.3f}  "
                      f"cont6/pair={ratio:.2f}", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(data.RESULTS_DIR, "exp3_statedecomp.csv")
    df.to_csv(out, index=False)
    print(f"[sd] wrote {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
