"""How many of TERRA's 256 latent dims carry the probed information?  Effective rank, then the same per-source probe
on the top-k principal components (basis fitted on clean + patch Z, no labels).  A first read only: truncating a
trained 256-d code is not the same as training with a k-d bottleneck (CoMo's 16-d), which changes what is learned."""
import numpy as np
from ..common.probes import per_source_readouts
from ..stage1_preserve.probes import pca_reduce
from .nmse_bootstrap import targets


def erank(Z):
    s = np.linalg.svd(Z - Z.mean(0), compute_uv=False) ** 2; p = s / s.sum()
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def main():
    for tag in ("tc_cont", "latp2_inv"):
        d = np.load(f"acdc_la/results/{tag}_val.npz", allow_pickle=True)
        Z, ZP = d["Z"].astype(np.float32), d["ZP"].astype(np.float32)
        s = np.linalg.svd(Z - Z.mean(0), compute_uv=False) ** 2; cum = np.cumsum(s) / s.sum()
        print(f"{tag}: dims {Z.shape[1]}, effective rank {erank(Z):.1f}; variance in top 16/32/64: "
              f"{cum[15]:.2f}/{cum[31]:.2f}/{cum[63]:.2f}", flush=True)
        for k in (16, 32, 64, 128, None):
            if k:
                both = pca_reduce(np.vstack([Z, ZP]), k); A, B = both[:len(Z)], both[len(Z):]
            else:
                A, B = Z, ZP
            P = per_source_readouts(A, targets(d["ACT"]), d["GE"], d["SI"], d["names"], extra={"net_patch": (B, "net", None)})
            r = {t: P[f"mean_r2_{t}"] for t in ("net", "seg1", "prof", "seq")}
            print(f"   k={str(k or Z.shape[1]):>4s} | NMSE net {1-r['net']:.3f} seg1 {1-r['seg1']:.3f} prof {1-r['prof']:.3f} "
                  f"seq {1-r['seq']:.3f} | drop {100*(1-P['mean_r2_net_patch']/P['mean_r2_net']):.1f}%", flush=True)


if __name__ == "__main__":
    main()
