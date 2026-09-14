"""Item 1b -- what does the LAM latent add BEYOND single-frame state?

The Exp-3 "VQ costs 32%/66%" headline measures the raw drop in z's action R^2
from emb (pre-VQ) to z_q (post-VQ).  But the state decomposition showed ~half of
emb's decodability is single-frame state -- which the decoder already has from
o_t.  So the raw drop conflates two things: VQ discarding redundant state (free)
and VQ discarding genuine transition (the real cost).

This separates them with takeaway 1's protocol applied to the LAM.  Give the
probe the single-frame patch feature f(o_t) (a strong state baseline: pg4,
12288-d, xyz R^2 ~0.70) and measure how much z adds ON TOP:

    increment(z) = R^2([f(o_t), z]) - R^2([f(o_t)])

for emb and z_q separately, per dim group (xyz / rot / gripper).  This is the
only thing a latent action should be judged on -- what it contributes beyond the
state the policy can already see.

Verdict:
  increment(z_q) ~= increment(emb)         -> VQ dropped redundant state, not
                                              transition; retract "VQ is the cost".
  increment(z_q) << increment(emb) (esp rot)-> VQ destroyed transition; original
                                              conclusion holds, now cleanly.

f(o_t) = pg4[i_g] from the same DINOv2 ViT-B/14 the LAM uses; z from exp3's
cache; pairs aligned by replaying exp3's loop (asserted).  GPU probes.
"""

import os
import numpy as np
import pandas as pd
import torch

import data
import probes_torch
from exp3_statedecomp import rebuild_pair_index

GROUPS = ("xyz", "rot", "cont6")


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"[inc] GPU {torch.cuda.get_device_name(0)}  free {free/2**30:.1f} GiB", flush=True)
    index = data.build_index()
    pg4 = np.load(os.path.join(data.CACHE_DIR, "pfeats_clean_g4.npy"))     # (nF, 12288)

    rows = []
    for stride in data.STRIDES:
        c = np.load(os.path.join(data.CACHE_DIR, f"exp3_z_s{stride}.npz"))
        emb, zq, Yc, Yg, val = c["Zc"], c["Zq"], c["Yc"], c["Yg"], c["val"].astype(bool)
        i_g, _, _ = rebuild_pair_index(index, stride)
        assert len(i_g) == len(Yc)
        yc_re = np.concatenate([data.episode_pairs(index, e, stride)[3]
                                for e in range(len(index["starts"]))
                                if len(data.episode_pairs(index, e, stride)[0])])
        assert np.allclose(yc_re, Yc, atol=1e-5), "pair order mismatch vs exp3 cache"

        S = pg4[i_g].astype(np.float32)                       # single-frame state
        tr, va = ~val, val

        def prb(F):
            r, _ = probes_torch.run_probe_set_t(F[tr], F[va], Yc[tr], Yc[va],
                                                Yg[tr], Yg[va], device=dev)
            return r

        base = prb(S)
        combos = {
            "S+emb": prb(np.concatenate([S, emb], 1)),
            "S+z_q": prb(np.concatenate([S, zq], 1)),
        }
        print(f"[inc] s={stride} baseline f(o_t) [{S.shape[1]}d]  "
              f"xyz={base['r2_xyz']:.3f} rot={base['r2_rot']:.3f} "
              f"cont6={base['r2_cont6']:.3f} grip={base['bal_acc_grip']:.3f}", flush=True)
        for name, r in combos.items():
            d = {g: r[f"r2_{g}"] - base[f"r2_{g}"] for g in GROUPS}
            dg = r["bal_acc_grip"] - base["bal_acc_grip"]
            print(f"[inc] s={stride} {name:6s}  "
                  f"xyz {r['r2_xyz']:.3f} (Δ{d['xyz']:+.3f})  "
                  f"rot {r['r2_rot']:.3f} (Δ{d['rot']:+.3f})  "
                  f"cont6 {r['r2_cont6']:.3f} (Δ{d['cont6']:+.3f})  "
                  f"grip {r['bal_acc_grip']:.3f} (Δ{dg:+.3f})", flush=True)
            for g in GROUPS:
                rows.append(dict(stride=stride, latent=name.split("+")[1],
                                 dim_group=g, base=base[f"r2_{g}"],
                                 combined=r[f"r2_{g}"], increment=d[g]))
            rows.append(dict(stride=stride, latent=name.split("+")[1],
                             dim_group="gripper", base=base["bal_acc_grip"],
                             combined=r["bal_acc_grip"], increment=dg))

    df = pd.DataFrame(rows)
    out = os.path.join(data.RESULTS_DIR, "exp3_incremental.csv")
    df.to_csv(out, index=False)
    print(f"[inc] wrote {out}")
    # headline: z_q increment as a fraction of emb increment, per dim
    piv = df.pivot_table(index=["stride", "dim_group"], columns="latent",
                         values="increment")
    piv["zq/emb"] = piv["z_q"] / piv["emb"]
    print(piv.to_string())


if __name__ == "__main__":
    main()
