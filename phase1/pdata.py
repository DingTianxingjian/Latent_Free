"""Phase-1 shared data: windows, frozen V-JEPA2 g4 tokens, eval targets, shuffles.

Both models (LAM baseline, PRLA) read from here — same windows, same split, same
frozen features, so the comparison is apples-to-apples.

Temporal unit (phase_1.md): a window of 4m+1 frames, m=1 -> span 4, 5 frames.
Target for *evaluation* is the net action over the window's 4m steps (build_pairs
with stride=4m gives exactly that: i, j=i+4m, y_cont = sum actions[i:i+4m]).

Frozen encoder: V-JEPA2 g4 (4x4 grid kept), cached per frame in
cache/vjfeats_clean_g4.npy as (N, 16*1024).  We view it as (N, 16, 1024): 16
spatial tokens of 1024-d per frame.  Global pooling is forbidden (costs ~50% of
Δf decodability, verified across encoders) — the layout is the point.
"""

import os

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data                                    # noqa: E402
import features_vjepa                          # noqa: E402  (GRID, G4, HID)

M = 1
SPAN = 4 * M                                   # 4 steps
NFRAMES = 4 * M + 1                            # 5 frames per window
NTOK = features_vjepa.G4 * features_vjepa.G4   # 16 spatial tokens (both encoders)

# Frozen-encoder substrates.  Both are a 4x4 g4 grid (16 tokens); only the hidden
# width differs.  The DINOv2 pg4 cache is exactly the substrate the OFFICIAL
# UniVLA LAM was built on, so `--encoder dino` isolates "substrate" from "data"
# when the from-scratch baseline fails.
ENCODERS = {
    "vjepa": dict(path="vjfeats_clean_g4.npy", hid=1024, stats="g4_stats_vjepa.npz"),
    "dino":  dict(path="pfeats_clean_g4.npy",  hid=768,  stats="g4_stats_dino.npz"),
}
HID = ENCODERS["vjepa"]["hid"]                 # default; train.py overrides per --encoder

# Bridge V2 substrate: V-JEPA2 g4 only (cache/bridge/vjfeats_g4.npy), same 16x1024
# layout as LIBERO's V-JEPA2 g4 — so the model/eval code is dataset-agnostic.
BRIDGE_DIR = os.path.join(data.CACHE_DIR, "bridge")
GRIP_THRESH = {"libero": 0.0, "bridge": 0.5, "bridge_big": 0.5}   # gripper binarization midpoint

_G4 = {}         # (dataset, encoder, standardize) -> array
_G4_RAW = {}     # (dataset, encoder) -> raw array


# Bridge variants: "bridge" is the original 622-episode subset (12 shards); the
# scaling curve showed it sits in a steeply-rising, degenerate regime, so
# "bridge_big" (5000 episodes / 187,587 frames, 200 shards) is the set method
# comparisons should use.  Both support either substrate; DINOv2 g4 is 16*768.
BRIDGE_DIRS = {"bridge": BRIDGE_DIR,
               "bridge_big": os.path.join(data.CACHE_DIR, "bridge_big")}
BRIDGE_FEATS = {"vjepa": ("vjfeats_clean_g4.npy", 1024),
                "dino":  ("dinofeats_clean_g4.npy", 768)}


def _g4_cfg(dataset, encoder):
    """(feature_path, hid, stats_path) for a substrate."""
    if dataset in BRIDGE_DIRS:
        d = BRIDGE_DIRS[dataset]
        fn, hid = BRIDGE_FEATS[encoder]
        # the original 622-ep cache predates the per-condition naming
        if dataset == "bridge" and encoder == "vjepa":
            fn = "vjfeats_g4.npy"
        return (os.path.join(d, fn), hid,
                os.path.join(data.CACHE_DIR, f"g4_stats_{dataset}_{encoder}.npz"))
    e = ENCODERS[encoder]
    return (os.path.join(data.CACHE_DIR, e["path"]), e["hid"],
            os.path.join(data.CACHE_DIR, e["stats"]))


def get_index(dataset="libero"):
    if dataset in BRIDGE_DIRS:
        z = np.load(os.path.join(BRIDGE_DIRS[dataset], "index.npz"))
        return {k: z[k] for k in z.files}
    return data.build_index()


def enc_hid(dataset, encoder):
    if dataset in BRIDGE_DIRS:
        return BRIDGE_FEATS[encoder][1]
    return ENCODERS[encoder]["hid"]


def _g4_stats(raw, index, stats_path):
    """Per-feature mean/std over TRAIN-split frames (no val leakage).  Cached.

    Raw g4 features are dominated by static appearance (std ~2.4) vs the tiny
    action-relevant frame-to-frame delta (~0.34); raw-MSE reconstruction chases
    appearance and starves the codes.  Whitening puts every dim on equal footing.
    """
    if os.path.exists(stats_path):
        s = np.load(stats_path)
        return s["mean"], s["std"]
    starts, lengths, is_val = index["starts"], index["lengths"], index["is_val"]
    train_rows = np.concatenate([np.arange(int(starts[e]), int(starts[e]) + int(lengths[e]))
                                 for e in range(len(starts)) if not bool(is_val[e])])
    flat = raw[train_rows].reshape(len(train_rows), -1)
    mean = flat.mean(0).astype(np.float32)
    std = flat.std(0).astype(np.float32)
    std[std < 1e-6] = 1.0
    np.savez(stats_path, mean=mean, std=std)
    return mean, std


def g4_tokens(dataset="libero", encoder="vjepa", standardize=True):
    """(N, 16, hid) per-frame frozen g4 tokens, cached in RAM.  Whitened by
    TRAIN-split stats when standardize=True (identical treatment across datasets)."""
    path, hid, stats_path = _g4_cfg(dataset, encoder)
    key = (dataset, encoder)
    if key not in _G4_RAW:
        arr = np.load(path)
        assert arr.shape[1] == NTOK * hid, (dataset, encoder, arr.shape)
        _G4_RAW[key] = arr
    raw = _G4_RAW[key]
    if not standardize:
        return raw.reshape(raw.shape[0], NTOK, hid)
    skey = (dataset, encoder, True)
    if skey not in _G4:
        mean, std = _g4_stats(raw, get_index(dataset), stats_path)
        _G4[skey] = ((raw - mean) / std).reshape(raw.shape[0], NTOK, hid).astype(np.float32)
    return _G4[skey]


def g4_corrupt(dataset, encoder, cond):
    """Corrupted-condition g4 tokens, whitened by the CLEAN TRAIN stats so every
    condition receives identical input normalisation (the convention the whole
    repo uses: only the pixels differ, never the scaling)."""
    d = BRIDGE_DIRS[dataset]
    stem = "vjfeats" if encoder == "vjepa" else "dinofeats"
    raw = np.load(os.path.join(d, f"{stem}_{cond}_g4.npy"), mmap_mode="r")
    _, hid, stats_path = _g4_cfg(dataset, encoder)
    mean, std = _g4_stats(g4_tokens(dataset, encoder, standardize=False)
                          .reshape(-1, NTOK * hid), get_index(dataset), stats_path)
    return ((np.asarray(raw) - mean) / std).reshape(-1, NTOK, hid).astype(np.float32)


def build_pairs(index, stride, split, dataset="libero"):
    """(i, j=i+stride, y_cont, y_grip) over episodes in `split`.  LIBERO defers to
    data.build_pairs (unchanged); Bridge uses the generic path over the index
    arrays with a dataset-appropriate gripper threshold."""
    if dataset == "libero":
        return data.build_pairs(index, stride, split)
    thr = GRIP_THRESH[dataset]
    want_val = split == "val"
    a = index["actions"]
    I, J, YC, YG = [], [], [], []
    for e in range(len(index["starts"])):
        if bool(index["is_val"][e]) != want_val:
            continue
        s, L = int(index["starts"][e]), int(index["lengths"][e])
        n = L - stride
        if n <= 0:
            continue
        i = s + np.arange(n)
        j = i + stride
        I.append(i); J.append(j)
        YC.append(np.stack([a[t:t + stride, :6].sum(0) for t in i]).astype(np.float32))
        YG.append((a[j - 1, 6] > thr).astype(np.int64))
    return (np.concatenate(I), np.concatenate(J), np.concatenate(YC), np.concatenate(YG))


def windows(index, split, dataset="libero"):
    """Window starts for a split.  Returns dict:
        i  (n,)   global row id of o_t (window first frame)
        j  (n,)   global row id of o_{t+span} (window last frame == target end)
        yc (n,6)  net continuous action over the span
        yg (n,)   gripper at span end, {0,1}
        frames (n, NFRAMES)  global row ids i..i+span (all same-episode)
    """
    i, j, yc, yg = build_pairs(index, SPAN, split, dataset)
    frames = i[:, None] + np.arange(NFRAMES)[None, :]
    assert (frames[:, -1] == j).all()          # window end == the target frame
    return dict(i=i, j=j, yc=yc, yg=yg, frames=frames)


def prla_windows(index, split, dataset="libero"):
    """PRLA positions: a 'current' frame o_t with a full HISTORY window
    [o_{t-span}..o_t] (span+1 frames) AND a full FUTURE window [o_{t+1}..o_{t+span}]
    (span frames), both inside the same episode.  History and future are kept as
    SEPARATE frame lists on purpose — they are encoded as separate clips so a
    bidirectional encoder's history tokens never see the future (causality).

    Target is the net action over [t, t+span] — the SAME target as the LAM
    window whose start is t, so the two models are comparable on the shared subset
    (PRLA drops the first `span` frames of each episode, where no history exists).

    Returns dict: t, i(=t), j(=t+span), hist (n,span+1), fut (n,span), yc, yg.
    """
    starts, lengths, is_val = index["starts"], index["lengths"], index["is_val"]
    actions = index["actions"]
    want_val = split == "val"
    ts = []
    for e in range(len(starts)):
        if bool(is_val[e]) != want_val:
            continue
        s, L = int(starts[e]), int(lengths[e])
        # local tau with tau-span >= 0 and tau+span <= L-1
        for tau in range(SPAN, L - SPAN):
            ts.append(s + tau)
    t = np.array(ts, np.int64)
    hist = t[:, None] + np.arange(-SPAN, 1)[None, :]        # [t-span .. t], span+1 frames
    fut = t[:, None] + np.arange(1, SPAN + 1)[None, :]      # [t+1 .. t+span], span frames
    yc = np.stack([actions[tt:tt + SPAN, :6].sum(0) for tt in t]).astype(np.float32)
    yg = np.array([(actions[tt + SPAN - 1, 6] > GRIP_THRESH[dataset]) for tt in t], np.int64)
    return dict(t=t, i=t, j=t + SPAN, hist=hist, fut=fut, yc=yc, yg=yg)


def _ep_of(index):
    """Map each global row id -> episode index (for same/different-episode shuffles)."""
    starts, lengths = index["starts"], index["lengths"]
    ep = np.empty(int(lengths.sum()), np.int64)
    for e, (s, L) in enumerate(zip(starts, lengths)):
        ep[s:s + L] = e
    return ep


def shuffle_starts(w, index, tier, seed=0):
    """Replacement window-start ids for the state-content control: keep o_t, swap
    the REST of the window for another window's frames.  Three OOD tiers:
      same_ep : a different window in the SAME episode (>= span away)
      diff_ep : a window from a DIFFERENT episode
      rand    : a fully random valid window start
    Returns (n, NFRAMES) global ids whose column 0 is the ORIGINAL o_t and whose
    columns 1.. come from the replacement window (so only the transition is
    corrupted, the anchor frame is intact).
    """
    rng = np.random.RandomState(seed)
    i = w["i"]
    n = len(i)
    ep = _ep_of(index)
    ep_i = ep[i]

    if tier == "rand":
        repl = i[rng.permutation(n)]
    elif tier == "diff_ep":
        repl = np.empty(n, np.int64)
        order = rng.permutation(n)
        # pair each window with another whose episode differs; fall back to roll
        for k in range(n):
            c = order[k]
            if ep_i[c] == ep_i[k]:
                c = order[(k + 1) % n]
            repl[k] = i[c]
    elif tier == "same_ep":
        # for each window, another window-start in the same episode, >= span away
        by_ep = {}
        for k in range(n):
            by_ep.setdefault(int(ep_i[k]), []).append(k)
        repl = i.copy()
        for e, idxs in by_ep.items():
            idxs = np.array(idxs)
            starts = i[idxs]
            for pos, k in enumerate(idxs):
                far = idxs[np.abs(starts - i[k]) >= SPAN]
                repl[k] = i[far[rng.randint(len(far))]] if len(far) else i[k]
    else:
        raise ValueError(tier)

    frames = repl[:, None] + np.arange(NFRAMES)[None, :]
    frames[:, 0] = i                            # keep the true anchor o_t
    return frames


def shuffle_perm(t, ep_ids, tier, seed=0):
    """Permutation of item indices for the PRLA future-shuffle control (keep the
    history, swap the future window for another item's future).  Three OOD tiers:
      same_ep : another item in the SAME episode, position >= span away
      diff_ep : an item from a DIFFERENT episode
      rand    : a fully random other item
    Returns an int array `perm` with fut_shuf = fut[perm]."""
    rng = np.random.RandomState(seed)
    n = len(t)
    if tier == "rand":
        return rng.permutation(n)
    if tier == "diff_ep":
        order = rng.permutation(n)
        perm = order.copy()
        for k in range(n):
            if ep_ids[order[k]] == ep_ids[k]:
                perm[k] = order[(k + 1) % n]
        return perm
    if tier == "same_ep":
        by_ep = {}
        for k in range(n):
            by_ep.setdefault(int(ep_ids[k]), []).append(k)
        perm = np.arange(n)
        for e, idxs in by_ep.items():
            idxs = np.array(idxs)
            for k in idxs:
                far = idxs[np.abs(t[idxs] - t[k]) >= SPAN]
                perm[k] = far[rng.randint(len(far))] if len(far) else k
        return perm
    raise ValueError(tier)


if __name__ == "__main__":
    idx = data.build_index()
    for sp in ("train", "val"):
        w = windows(idx, sp)
        print(f"{sp}: {len(w['i'])} windows, frames {w['frames'].shape}, "
              f"yc {w['yc'].shape}")
    G = g4_tokens()
    print(f"g4 tokens {G.shape} dtype {G.dtype}")
    w = windows(idx, "val")
    for tier in ("same_ep", "diff_ep", "rand"):
        fr = shuffle_starts(w, idx, tier, seed=0)
        anchor_ok = (fr[:, 0] == w["i"]).all()
        moved = (fr[:, 1:] != w["frames"][:, 1:]).any(1).mean()
        print(f"  shuffle {tier:8s}: anchor kept {anchor_ok}, frac windows moved {moved:.2f}")
