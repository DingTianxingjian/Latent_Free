"""Verification for the Exp-2 Part-B subspace result.

Part B reports that the top-k PCs of Delta-f retain almost none of the clean
action subspace -- far *below* the k/D level a random subspace would give.  That
is surprising enough to check two ways before it is believed:

1.  Is it an artifact of the standardize-then-unstandardize round trip?
    The probe is fitted on standardized features, so the raw-space weight is
    w_raw = w_std / x_sd.  Coordinates with a small std get amplified, which
    tilts the "action direction" toward low-variance coordinates -- and those
    tend to sit in the trailing PCs.  So we recompute the action basis from a
    ridge fitted *directly on raw features*, where no such amplification exists,
    and compare.

2.  The decisive, metric-free test: actually build the bottleneck.
    Project Delta-f onto its own top-k PCs, refit the probe on the projection,
    and measure val R^2.  This asks the question the experiment is really about
    -- "if a LAM keeps only the top-k variance directions, how much action
    information survives?" -- without depending on any subspace-overlap metric.
"""

import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import data
import probes
from features import CONDITIONS

KS = (1, 2, 4, 8, 16, 32, 64, 128)


def raw_action_basis(Xtr, Ytr, alphas=probes.RIDGE_ALPHAS):
    """Ridge fitted directly on raw (unstandardized) Delta-f -> orthonormal basis."""
    best, bw = None, None
    for a in alphas:
        m = Ridge(alpha=a, fit_intercept=True, solver="cholesky").fit(Xtr, Ytr)
        s = float(np.mean(r2_score(Ytr, m.predict(Xtr), multioutput="raw_values")))
        if best is None or s > best:
            best, bw = s, m.coef_
    Q, _ = np.linalg.qr(bw.T)
    return Q


def energy(Pk, A):
    return float(np.sum((Pk @ A) ** 2) / A.shape[1])


def main():
    index = data.build_index()
    rows = []
    for stride in data.STRIDES:
        itr, jtr, yc_tr, _ = data.build_pairs(index, stride, "train")
        iva, jva, yc_va, _ = data.build_pairs(index, stride, "val")

        clean = np.load(os.path.join(data.CACHE_DIR, "feats_clean.npy"))
        Dtr_c, Dva_c = clean[jtr] - clean[itr], clean[jva] - clean[iva]
        # (1) two ways of defining the clean action subspace
        _, W_std = probes.ridge_probe(Dtr_c, yc_tr, Dva_c, yc_va)
        A_std = np.linalg.qr(W_std.T)[0]
        A_raw = raw_action_basis(Dtr_c, yc_tr)
        prin = np.linalg.svd(A_std.T @ A_raw, compute_uv=False)
        print(f"[verify] s={stride} cos(principal angles) between the two action "
              f"bases: {np.round(prin, 3)}")
        del clean

        for cond in CONDITIONS:
            p = os.path.join(data.CACHE_DIR, f"feats_{cond}.npy")
            if not os.path.exists(p):
                continue
            f = np.load(p)
            Dtr, Dva = f[jtr] - f[itr], f[jva] - f[iva]
            pca = PCA(n_components=max(KS), svd_solver="full", random_state=0).fit(Dtr)
            evr = np.cumsum(pca.explained_variance_ratio_)
            Ztr_all, Zva_all = pca.transform(Dtr), pca.transform(Dva)
            for k in KS:
                Pk = pca.components_[:k]
                # (2) the decisive test: refit the probe inside the bottleneck
                res, _ = probes.ridge_probe(Ztr_all[:, :k], yc_tr,
                                            Zva_all[:, :k], yc_va)
                rows.append(dict(
                    stride=stride, condition=cond, k=k,
                    cum_var_explained=float(evr[k - 1]),
                    energy_std_basis=energy(Pk, A_std),
                    energy_raw_basis=energy(Pk, A_raw),
                    chance_energy=k / Dtr.shape[1],
                    r2_xyz_topk=res["r2_xyz"], r2_cont6_topk=res["r2_cont6"],
                ))
                if k in (8, 32, 128):
                    r = rows[-1]
                    print(f"[verify] s={stride} {cond:14s} k={k:3d} "
                          f"var={r['cum_var_explained']:.3f} "
                          f"E_std={r['energy_std_basis']:.4f} "
                          f"E_raw={r['energy_raw_basis']:.4f} "
                          f"chance={r['chance_energy']:.4f} "
                          f"R2xyz(top-k)={r['r2_xyz_topk']:.3f}", flush=True)
            del f, Dtr, Dva, Ztr_all, Zva_all
    df = pd.DataFrame(rows)
    out = os.path.join(data.RESULTS_DIR, "exp2_verify_topk.csv")
    df.to_csv(out, index=False)
    print(f"[verify] wrote {out}")


if __name__ == "__main__":
    main()
