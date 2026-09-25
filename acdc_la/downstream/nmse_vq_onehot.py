"""Read the VQ tokenizer's codes as categories (one-hot per token) instead of as codebook vectors."""
import numpy as np
from acdc_la.common.probes import per_source_readouts
from acdc_la.downstream.nmse_bootstrap import targets
d = np.load("acdc_la/results/tc_vq256_val.npz", allow_pickle=True)
Z, ZP = d["Z"].astype(np.float32), d["ZP"].astype(np.float32); n = len(Z)
def onehot(A, tables):
    T = A.reshape(n, 16, -1); cols = []
    for t in range(16):
        tab = tables[t]                                                   # (K, 64) codebook entries seen on clean Z
        dist = ((T[:, t, None, :] - tab[None]) ** 2).sum(-1); idx = dist.argmin(1)   # nearest entry (exact on clean)
        oh = np.zeros((n, len(tab)), np.float32); oh[np.arange(n), idx] = 1; cols.append(oh)
    return np.concatenate(cols, 1)
tables = [np.unique(np.round(Z.reshape(n, 16, -1)[:, t], 4), axis=0) for t in range(16)]
X, XP = onehot(Z, tables), onehot(ZP, tables)
for name, A, B in (("codebook vectors (as dumped)", Z, ZP), ("one-hot per token", X, XP)):
    P = per_source_readouts(A, targets(d["ACT"]), d["GE"], d["SI"], d["names"], extra={"net_patch": (B, "net", None)})
    r = {k: P[f"mean_r2_{k}"] for k in ("net", "seg1", "prof", "seq")}
    print(f"{name:30s} dims {A.shape[1]:5d} | NMSE net {1-r['net']:.3f} seg1 {1-r['seg1']:.3f} prof {1-r['prof']:.3f} "
          f"seq {1-r['seq']:.3f} | drop {100*(1-P['mean_r2_net_patch']/P['mean_r2_net']):.1f}%", flush=True)
