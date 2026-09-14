"""Exp 2B on patch tokens -- the theory test, moved out of CLS space.

The CLS Exp 2B (SUMMARY, takeaway 2) found the action subspace sits *below*
chance inside the top-k PCs of Delta-f.  But Exp 4 then showed the CLS token
throws away most of what Delta-f knows (stride-1 xyz R^2 0.194 -> 0.612 on the
4x4 patch grid), so that magnitude is a CLS-space statement.  This re-runs the
identical Part-B math where the action signal actually lives:

    space   pmean : mean over the 16x16 patch grid          768-d  (CLS width)
            pg4   : 4x4 spatial pooling of the patch grid  12288-d  (Exp-4 space)
            cls   : the original, recomputed for a side-by-side reference

For each we ask the same two questions as exp2_verify.py, on the CLEAN condition:

  (1) subspace overlap  ||P_k A||_F^2 / 6   vs the chance level k / D
      A = clean action subspace (span of the 6 ridge weight rows, orthonormal),
      computed two ways (standardized- and raw-feature ridge) to rule out the
      standardize-then-unstandardize amplification artifact.
  (2) the decisive, metric-free test: project Delta-f onto its own top-k PCs,
      refit the probe, and read val R^2.  "% of full" = that / the full-D probe.

Only the clean patch features are cached; distractor conditions would need a GPU
extraction pass and are left as the follow-up.  CPU only -- pure linear algebra.
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import data
import probes
from exp2_verify import raw_action_basis, energy

# pg4 is 12288-d, so full SVD is wasteful; randomized is exact enough for the
# top components and still returns explained_variance_ratio_ against total var.
SPACES = {
    "cls":   ("feats_clean.npy",        768,   "full"),
    "pmean": ("pfeats_clean_mean.npy",  768,   "full"),
    "pg4":   ("pfeats_clean_g4.npy",    12288, "randomized"),
}
KS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
REPORT_KS = (8, 32, 128)


def run_space(name, fname, D, solver, index):
    path = os.path.join(data.CACHE_DIR, fname)
    if not os.path.exists(path):
        print(f"[2Bp] {name}: {path} missing -- skipping", flush=True)
        return []
    feats = np.load(path)
    rows = []
    for stride in data.STRIDES:
        t0 = time.time()
        itr, jtr, yc_tr, _ = data.build_pairs(index, stride, "train")
        iva, jva, yc_va, _ = data.build_pairs(index, stride, "val")
        Dtr = (feats[jtr] - feats[itr]).astype(np.float32)
        Dva = (feats[jva] - feats[iva]).astype(np.float32)

        # clean action subspace, both ways, + the full-D probe (the "100%" ref)
        full, W_std = probes.ridge_probe(Dtr, yc_tr, Dva, yc_va)
        A_std = np.linalg.qr(W_std.T)[0]
        A_raw = raw_action_basis(Dtr, yc_tr)
        prin = np.linalg.svd(A_std.T @ A_raw, compute_uv=False)

        ncomp = min(max(KS), D - 1)
        pca = PCA(n_components=ncomp, svd_solver=solver, random_state=0).fit(Dtr)
        evr = np.cumsum(pca.explained_variance_ratio_)
        Ztr, Zva = pca.transform(Dtr), pca.transform(Dva)

        print(f"[2Bp] {name:5s} s={stride}  D={D}  full xyz R2={full['r2_xyz']:.3f}"
              f"  cont6={full['r2_cont6']:.3f}  alpha={full['alpha']:g}"
              f"  cos(A_std,A_raw)={np.round(prin, 3)}  [{time.time()-t0:.0f}s]",
              flush=True)

        for k in KS:
            if k >= D:
                continue
            Pk = pca.components_[:k]
            res, _ = probes.ridge_probe(Ztr[:, :k], yc_tr, Zva[:, :k], yc_va)
            r = dict(
                space=name, stride=stride, condition="clean", k=k, D=D,
                cum_var_explained=float(evr[k - 1]),
                energy_std_basis=energy(Pk, A_std),
                energy_raw_basis=energy(Pk, A_raw),
                chance_energy=k / D,
                r2_xyz_topk=res["r2_xyz"], r2_cont6_topk=res["r2_cont6"],
                r2_xyz_full=full["r2_xyz"], r2_cont6_full=full["r2_cont6"],
                frac_of_full_xyz=res["r2_xyz"] / full["r2_xyz"]
                if full["r2_xyz"] > 1e-6 else float("nan"),
            )
            rows.append(r)
            if k in REPORT_KS:
                print(f"[2Bp] {name:5s} s={stride} k={k:3d}  var={r['cum_var_explained']:.3f}"
                      f"  E={r['energy_std_basis']:.4f} (raw {r['energy_raw_basis']:.4f})"
                      f"  chance={r['chance_energy']:.4f}"
                      f"  E/chance={r['energy_std_basis']/r['chance_energy']:.2f}x"
                      f"  R2xyz(top-k)={r['r2_xyz_topk']:.3f}"
                      f"  ({r['frac_of_full_xyz']*100:.0f}% of full)", flush=True)
        del Dtr, Dva, Ztr, Zva, pca
    del feats
    return rows


def main():
    index = data.build_index()
    want = os.environ.get("LF_2BP_SPACES")          # e.g. "cls" to smoke-test
    names = want.split(",") if want else list(SPACES)
    rows = []
    for name in names:
        fname, D, solver = SPACES[name]
        rows += run_space(name, fname, D, solver, index)
    df = pd.DataFrame(rows)
    out = os.path.join(data.RESULTS_DIR, "exp2b_patch_pca.csv")
    df.to_csv(out, index=False)
    print(f"[2Bp] wrote {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
