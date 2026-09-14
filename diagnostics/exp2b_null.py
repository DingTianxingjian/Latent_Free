"""Exp 2B null control -- does PCA *ordering* carry action, or is 'leads' empty?

Exp 2B on patch tokens reported that the top-k PCs of Delta-f recover action
*faster* than they recover variance ("action leads variance").  But that
compared a top-k action curve against the *variance* curve.  The honest null is
a random k-dim subspace: if action is spread isotropically, then

    top-k  ~=  random-k  ~=  bottom-k

and "leads/lags" is an artifact of comparing to the variance curve -- the PCA
ordering would carry no action information at all.  This decides which of three
write-ups is true, per space:

    top-k >> random-k ~ bottom-k   -> action in HIGH-variance PCs   (leads, real)
    top-k << random-k              -> action in the low-variance TAIL (CLS story)
    top-k ~= random-k ~= bottom-k  -> PCA ordering is uninformative  (neither)

Metric: the same project -> refit -> val R^2 as exp2b_patch, on CLEAN Delta-f.
SVD and projections run on GPU (the ai node's GPU is reserved anyway); the refit
is the identical sklearn probe, so the numbers line up with the 2B table.
"""

import os
import time

import numpy as np
import pandas as pd
import torch

import data
import probes

SPACES = {
    "cls":   ("feats_clean.npy",        768),
    "pmean": ("pfeats_clean_mean.npy",  768),
    "pg4":   ("pfeats_clean_g4.npy",    12288),
}
KS = (8, 16, 32, 64, 128, 256, 512)
N_SEED = 5


def refit(Ztr, Zva, yc_tr, yc_va):
    """% of full is applied by the caller; here just the two R^2 we report."""
    res, _ = probes.ridge_probe(Ztr, yc_tr, Zva, yc_va)
    return res["r2_xyz"], res["r2_cont6"]


def run(name, fname, D, index, dev):
    path = os.path.join(data.CACHE_DIR, fname)
    if not os.path.exists(path):
        print(f"[null] {name}: {path} missing -- skip", flush=True)
        return []
    feats = np.load(path)
    rows = []
    for stride in data.STRIDES:
        t0 = time.time()
        itr, jtr, yc_tr, _ = data.build_pairs(index, stride, "train")
        iva, jva, yc_va, _ = data.build_pairs(index, stride, "val")
        Dtr = (feats[jtr] - feats[itr]).astype(np.float32)
        Dva = (feats[jva] - feats[iva]).astype(np.float32)

        xyz_full, cont6_full = refit(Dtr, Dva, yc_tr, yc_va)

        # SVD on GPU -> full principal-component ordering (for top-k and bottom-k)
        Gtr = torch.from_numpy(Dtr).to(dev)
        Gva = torch.from_numpy(Dva).to(dev)
        mu = Gtr.mean(0, keepdim=True)
        _, _, Vh = torch.linalg.svd(Gtr - mu, full_matrices=False)   # Vh: (D,D)

        def project_refit(Q):                       # Q: (D,k) orthonormal columns
            Ztr = (Gtr @ Q).cpu().numpy()
            Zva = (Gva @ Q).cpu().numpy()
            return refit(Ztr, Zva, yc_tr, yc_va)

        for k in KS:
            if k > D:
                continue
            top_xyz, _ = project_refit(Vh[:k].T)
            bot_xyz, _ = project_refit(Vh[-k:].T)
            rand = []
            for s in range(N_SEED):
                g = torch.randn(D, k, generator=torch.Generator(dev).manual_seed(s),
                                device=dev)
                Q, _ = torch.linalg.qr(g)           # (D,k) orthonormal
                rand.append(project_refit(Q)[0])
            rand = np.array(rand)
            rows.append(dict(
                space=name, stride=stride, k=k, D=D,
                r2_xyz_full=xyz_full,
                r2_xyz_top=top_xyz, r2_xyz_bottom=bot_xyz,
                r2_xyz_rand_mean=float(rand.mean()),
                r2_xyz_rand_std=float(rand.std()),
                frac_top=top_xyz / xyz_full if xyz_full > 1e-6 else float("nan"),
                frac_rand=float(rand.mean()) / xyz_full if xyz_full > 1e-6 else float("nan"),
                frac_bottom=bot_xyz / xyz_full if xyz_full > 1e-6 else float("nan"),
            ))
            r = rows[-1]
            verdict = ("TOP>rand" if top_xyz > r["r2_xyz_rand_mean"] + 2 * r["r2_xyz_rand_std"]
                       else "TOP<rand" if top_xyz < r["r2_xyz_rand_mean"] - 2 * r["r2_xyz_rand_std"]
                       else "top~rand")
            print(f"[null] {name:5s} s={stride} k={k:3d}  "
                  f"top={top_xyz:.3f} ({r['frac_top']*100:.0f}%)  "
                  f"rand={r['r2_xyz_rand_mean']:.3f}±{r['r2_xyz_rand_std']:.3f} "
                  f"({r['frac_rand']*100:.0f}%)  bottom={bot_xyz:.3f}  -> {verdict}",
                  flush=True)
        print(f"[null] {name} s={stride} done  full xyz={xyz_full:.3f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        del Gtr, Gva, Vh, Dtr, Dva
        if dev == "cuda":
            torch.cuda.empty_cache()
    del feats
    return rows


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[null] device={dev}  seeds={N_SEED}  ks={KS}", flush=True)
    index = data.build_index()
    want = os.environ.get("LF_NULL_SPACES")          # e.g. "cls" to smoke-test
    names = want.split(",") if want else list(SPACES)
    rows = []
    for name in names:
        fname, D = SPACES[name]
        rows += run(name, fname, D, index, dev)
    df = pd.DataFrame(rows)
    out = os.path.join(data.RESULTS_DIR, "exp2b_null.csv")
    df.to_csv(out, index=False)
    print(f"[null] wrote {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
