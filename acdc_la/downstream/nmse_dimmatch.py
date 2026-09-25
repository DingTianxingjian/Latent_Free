import numpy as np, os
from acdc_la.common.probes import per_source_readouts
from acdc_la.stage1_preserve.probes import pca_reduce
from acdc_la.downstream.nmse_bootstrap import targets
R = "acdc_la/results"
for tag, k in (("latp2_inv", None), ("univla_noinh", 256), ("univla_noinh", 64), ("univla_matched", 256)):
    d = np.load(f"{R}/{tag}_val.npz", allow_pickle=True)
    Z, ZP = d["Z"].astype(np.float32), d["ZP"].astype(np.float32)
    if k:                                                    # one unsupervised basis, fitted on clean Z, applied to both
        both = pca_reduce(np.vstack([Z, ZP]), k); Z, ZP = both[:len(Z)], both[len(Z):]
    P = per_source_readouts(Z, targets(d["ACT"]), d["GE"], d["SI"], d["names"], extra={"net_patch": (ZP, "net", None)})
    r = {t: P[f"mean_r2_{t}"] for t in ("net", "seg1", "prof", "seq")}
    drop = 100 * (1 - P["mean_r2_net_patch"] / P["mean_r2_net"])
    print(f"{tag:15s} dims {Z.shape[1]:5d} | NMSE net {1-r['net']:.3f}  seg1 {1-r['seg1']:.3f}  prof {1-r['prof']:.3f}  seq {1-r['seq']:.3f} | drop {drop:.1f}%", flush=True)
