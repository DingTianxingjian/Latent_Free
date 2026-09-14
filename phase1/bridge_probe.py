"""Bridge V2 subset: run the EXISTING probing protocol (single / diff / pair /
pair_shuf, per-axis) on frozen V-JEPA2 g4 features — the same diagnostic we ran on
LIBERO, now on multi-operator REAL teleoperation data.

Produces the two numbers we predicted but never measured:
  * state-confusion ratio  = pair_shuf / pair  (how much a two-frame readout is
    just re-encoded o_t state)  — on scripted LIBERO this was ~1.0 (huge);
  * transition budget       = pair - pair_shuf (genuine two-frame contribution).

Bridge action is 7-d [xyz(3), rot(3), gripper(1)], same as LIBERO, so the probe
groups transfer.  Bridge is real teleop (~5 Hz, variable), so 'stride' is a
coarser time gap than LIBERO's; we report stride 1 and 4 and flag the caveat.

GPU job: extracts V-JEPA2 g4 for the bridge subset (cached) then probes.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data                                    # noqa: E402
import probes_torch                            # noqa: E402
import features_vjepa                          # noqa: E402

BDIR = os.path.join(data.CACHE_DIR, "bridge")
FEAT = os.path.join(BDIR, "vjfeats_g4.npy")
STRIDES = (1, 4)


def load_index():
    z = np.load(os.path.join(BDIR, "index.npz"))
    return dict(starts=z["starts"], lengths=z["lengths"], is_val=z["is_val"], actions=z["actions"])


def build_pairs(index, stride, split):
    want_val = split == "val"
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
        a = index["actions"]
        yc = np.stack([a[t:t + stride, :6].sum(0) for t in i]).astype(np.float32)
        yg = (a[j - 1, 6] > 0.5).astype(np.int64)
        YC.append(yc); YG.append(yg)
    return (np.concatenate(I), np.concatenate(J),
            np.concatenate(YC), np.concatenate(YG))


@torch.no_grad()
def extract(index, device):
    if os.path.exists(FEAT):
        print(f"[bridge] {FEAT} cached", flush=True)
        return
    frames = np.load(os.path.join(BDIR, "frames.npy"))         # (N,256,256,3) u8
    model = features_vjepa.load_encoder(device)
    N = len(frames)
    out = np.zeros((N, features_vjepa.DIMS["g4"]), np.float32)
    t0 = time.time()
    # extract per episode span to bound memory; encode_frames resizes 256->256 (no-op)
    step = 512
    for k in range(0, N, step):
        z = features_vjepa.encode_frames(model, frames[k:k + step], device)
        out[k:k + step] = z["g4"]
        if (k // step) % 20 == 0:
            print(f"  [bridge:extract] {k+step}/{N} [{time.time()-t0:.0f}s]", flush=True)
    np.save(FEAT, out)
    print(f"[bridge] saved {FEAT} {out.shape}", flush=True)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    index = load_index()
    print(f"[bridge] {len(index['lengths'])} episodes, {int(index['lengths'].sum())} frames, "
          f"{int(index['is_val'].sum())} val", flush=True)
    extract(index, dev)
    feats = np.load(FEAT)

    rows = []
    for stride in STRIDES:
        itr, jtr, yctr, ygtr = build_pairs(index, stride, "train")
        iva, jva, ycva, ygva = build_pairs(index, stride, "val")
        rng = np.random.RandomState(stride)
        ptr, pva = rng.permutation(len(jtr)), rng.permutation(len(jva))

        def feat(kind, i, j, p):
            if kind == "single":    return feats[i]
            if kind == "diff":      return feats[j] - feats[i]
            if kind == "pair":      return np.concatenate([feats[i], feats[j]], 1)
            if kind == "pair_shuf": return np.concatenate([feats[i], feats[j][p]], 1)

        res = {}
        for kind in ("single", "diff", "pair", "pair_shuf"):
            Ftr = feat(kind, itr, jtr, ptr).astype(np.float32)
            Fva = feat(kind, iva, jva, pva).astype(np.float32)
            r, _ = probes_torch.run_probe_set_t(Ftr, Fva, yctr, ycva, ygtr, ygva, device=dev)
            res[kind] = r
            rows.append(dict(dataset="bridge", stride=stride, feature=kind, dim=Ftr.shape[1],
                             r2_xyz=r["r2_xyz"], r2_rot=r["r2_rot"], r2_cont6=r["r2_cont6"],
                             bal_acc_grip=r["bal_acc_grip"], n_train=len(itr), n_val=len(iva)))
            print(f"[bridge] s={stride} {kind:9s} d={Ftr.shape[1]:5d} xyz={r['r2_xyz']:.3f} "
                  f"rot={r['r2_rot']:.3f} cont6={r['r2_cont6']:.3f} grip={r['bal_acc_grip']:.3f}",
                  flush=True)
        # headline diagnostics
        sc = res["pair_shuf"]["r2_cont6"] / res["pair"]["r2_cont6"] if res["pair"]["r2_cont6"] > 1e-6 else float("nan")
        tb = res["pair"]["r2_cont6"] - res["pair_shuf"]["r2_cont6"]
        rows.append(dict(dataset="bridge", stride=stride, feature="SUMMARY", dim=0,
                         state_confusion_ratio_cont6=sc, transition_budget_cont6=tb))
        print(f"[bridge] s={stride} STATE-CONFUSION ratio(cont6)={sc:.3f}  TRANSITION budget(cont6)={tb:.3f}",
              flush=True)

    os.makedirs(os.path.join(data.RESULTS_DIR, "phase1"), exist_ok=True)
    out = os.path.join(data.RESULTS_DIR, "phase1", "bridge_probe.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[bridge] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
