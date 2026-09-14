"""Item 4a -- is the VQ cost real, or is the linear probe penalizing VQ geometry?

Exp 3 found z_q (post-VQ) decodes action far worse than emb (pre-VQ): the
quantization step is lossy.  But z_q is probed linearly, and a linear probe
could be unfair to the discrete codebook's geometry.  The codebook-geometry-free
upper bound is: one-hot the 4 code indices and regress action from that (64-d).
If one-hot ~= z_q, the VQ cost is real; if one-hot >> z_q, we overstated it.

(Mathematically z_q = onehot @ codebook is a linear image of the one-hot, so a
linear probe on z_q cannot exceed the one-hot probe -- this checks the converse,
that it does not fall short either, i.e. the codebook embedding loses no
decodable information.)  Uses only exp3's cached code indices; CPU, seconds.
"""
import os
import numpy as np
import pandas as pd
import probes
import data

rows = []
for s in data.STRIDES:
    d = np.load(os.path.join(data.CACHE_DIR, f"exp3_z_s{s}.npz"))
    IDX, Yc, Yg, val = d["IDX"], d["Yc"], d["Yg"], d["val"].astype(bool)
    K = int(IDX.max()) + 1
    oh = np.eye(K, dtype=np.float32)[IDX].reshape(len(IDX), -1)   # (N, num_codes*K)
    res, _ = probes.run_probe_set(oh[~val], oh[val], Yc[~val], Yc[val],
                                  Yg[~val], Yg[val])
    print(f"[onehot] s={s} ({oh.shape[1]}-d)  xyz={res['r2_xyz']:.3f} "
          f"rot={res['r2_rot']:.3f} cont6={res['r2_cont6']:.3f} "
          f"grip={res['bal_acc_grip']:.3f}", flush=True)
    rows.append(dict(stride=s, feature="onehot_codes", dim=oh.shape[1],
                     r2_xyz=res["r2_xyz"], r2_rot=res["r2_rot"],
                     r2_cont6=res["r2_cont6"], bal_acc_grip=res["bal_acc_grip"],
                     alpha=res["alpha"], C=res["C"]))

pd.DataFrame(rows).to_csv(os.path.join(data.RESULTS_DIR, "exp3_onehot.csv"), index=False)
print("[onehot] wrote results/exp3_onehot.csv")
