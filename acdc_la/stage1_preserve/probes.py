"""Stage I metrics -- does the tokenizer PRESERVE the temporal structure of the effect?  All offline, CPU,
from a validation dump (this repo's or a legacy swaplam readout).

A. Model-free, on the frozen quadrant-pooled difference sequence dq (K,4,768) of the val clips:
   for each temporal subset S in {DC} (= mean-pooled = endpoint), {DC,AC1}, {DC,AC1,AC2}, all K modes:
     temporal_var_kept[S]   1 - ||dq - proj_S dq||^2 / ||dq||^2         what the operator keeps of the sequence
     r2_<target>[S]         per-source ridge cells_S -> action target, targets
                              net  = sum_k a_k (3)      seg1 = a_0 (3)      prof = a - mean_k a (K*3, shape only)
                              seq  = a (K*3, the whole per-segment sequence)
   This is the field's problem measured directly: mean pooling / endpoint differencing keep DC and lose
   every non-DC target (prof, seq) by construction; the table says how much, at what dimensionality.
B. On the tokenizer's code z (n_cells x d):  the same four targets, the moving-patch drop (same probe read on
   the patch rendering), and
     rev_cos[mode]         cos( z_cell(reversed clip), s_mode * z_cell(clip) ) with s from the basis parity
                            (DC flips sign, AC1 is invariant, AC2 flips).  For an invariant mode the cells are IDENTICAL
                            under reversal, so its rev_cos is 1 by construction and carries no information; the number
                            that matters is the sign-flipping modes' (is E odd on the DC cells?) and r2_rev_net below.
     r2_rev_net            the net-action reader applied to z(reversed) against -net: reversal consistency in action units.
Usage:  python -m acdc_la.stage1_preserve.probes rec_src s1_acdc ...   (tags or paths) -> results/stage1_probes.json
"""
import sys, json, os, math
import numpy as np

from ..paths import RESULTS
from ..common import readers as R
from ..common.probes import per_source_readouts

SUBSETS = {"dc": [0], "dc_ac1": [0, 1], "dc_ac1_ac2": [0, 1, 2]}


def dct_rows(K, n):
    k = np.arange(K); rows = [np.ones(K)]
    for m in range(1, n): rows.append(math.sqrt(2.0 / K) * np.cos(math.pi * m * (k + 0.5) / K))
    return np.stack(rows)                                     # (n,K), DC un-normalised (sum), AC orthonormal


def cells_np(dq, rows):
    """dq (n,K,4,768), rows (C,K) -> (n, 4*C*768) ordered [q, c]."""
    c = np.einsum("ck,nkqd->nqcd", rows, dq)
    return c.reshape(len(dq), -1)


def targets_of(d):
    t = dict(net=d["NET"], seg1=d["SEG1"])
    if "PROF" in d: t.update(prof=d["PROF"], seq=d["SEQ"])
    return t


def pca_reduce(X, k, seed=0):
    """Unsupervised PCA of X (n,D) to k dims, fitted on all clips (no labels involved).  Equalises the DIMENSION of the
    temporal subsets before the ridge, so a 12,288-d subset is not penalised for its size at n ~ 270 train clips."""
    X = np.asarray(X, np.float32); mu = X.mean(0); Xc = X - mu
    rs = np.random.RandomState(seed); Q = rs.randn(X.shape[1], k + 16).astype(np.float32)          # randomized PCA (Halko)
    Y = Xc @ Q
    for _ in range(3): Q, _ = np.linalg.qr(Xc.T @ Y); Y = Xc @ Q
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    return (Y @ Vt.T)[:, :k]


def model_free(d, pca_dims=(256,)):
    """Part A on the frozen dq of the dump.  Each subset is probed at its raw dimension AND after PCA to `pca_dims`
    (the tokenizer's z has 256 dims), because the raw ridge at n ~ 270 favours the smallest subset by variance alone."""
    n = len(d["NET"]); K = int(d["K"]) if "K" in d else d["DF"].shape[1] // (4 * 768)
    dq = d["DF"].astype(np.float32).reshape(n, K, 4, 768)
    full = dct_rows(K, K); un = full / np.linalg.norm(full, axis=1, keepdims=True)    # orthonormal for projections
    coef = np.einsum("ck,nkqd->ncqd", un, dq)                                            # (n,K,4,768)
    tot = (dq ** 2).sum((1, 2, 3)); out = {}
    subsets = {**{k: v for k, v in SUBSETS.items() if max(v) < K}, "full": list(range(K))}
    for name, S in subsets.items():
        kept = (coef[:, S] ** 2).sum((1, 2, 3)) / np.maximum(tot, 1e-9)
        rows = {}
        for s, tr, te in R.source_splits(d["GE"], d["SI"]):
            rows[d["names"][s]] = dict(n_test=int(te.sum()), temporal_var_kept=float(kept[te].mean()))
        X = cells_np(dq, full[S])
        P = per_source_readouts(X, targets_of(d), d["GE"], d["SI"], d["names"])
        agg = R.aggregate(rows)
        out[name] = dict(dims=int(X.shape[1]), n_modes=len(S), temporal_var_kept=agg.get("temporal_var_kept"),
                         **{k: v for k, v in P.items() if k.startswith(("mean_r2", "umean_r2", "n_sources"))})
        for k in pca_dims:
            if k >= X.shape[1]: continue
            Pk = per_source_readouts(pca_reduce(X, k), targets_of(d), d["GE"], d["SI"], d["names"])
            out[f"{name}@pca{k}"] = dict(dims=k, n_modes=len(S), temporal_var_kept=agg.get("temporal_var_kept"),
                                         **{kk: v for kk, v in Pk.items() if kk.startswith(("mean_r2", "umean_r2", "n_sources"))})
    return out


def on_code(d):
    """Part B on the tokenizer code."""
    Z = d["Z"].astype(np.float64); n = len(Z); n_cells = int(d["n_cells"]) if "n_cells" in d else 8
    dcode = Z.shape[1] // n_cells; n_modes = n_cells // 4
    extra = {"net_patch": (d["ZP"], "net", None)} if "ZP" in d else {}
    P = per_source_readouts(Z, targets_of(d), d["GE"], d["SI"], d["names"], extra=extra)
    out = dict(dims=int(Z.shape[1]), n_cells=n_cells, **{k: v for k, v in P.items() if k.startswith(("mean_r2", "umean_r2", "n_sources"))})
    rc, rp = out.get("mean_r2_net", np.nan), out.get("mean_r2_net_patch", np.nan)
    out["patch_drop_pct"] = float(100 * (1 - rp / rc)) if rc > 0.02 else float("nan")
    if "ZREV" in d:
        K = int(d["K"]); rows = dct_rows(K, n_modes); flipped = rows[:, ::-1]
        par = np.sign((rows * flipped).sum(1)); signs = -par                              # DC -> -1, AC1 -> +1, AC2 -> -1
        Zc, Rc = Z.reshape(n, 4, n_modes, dcode), d["ZREV"].astype(np.float64).reshape(n, 4, n_modes, dcode)
        for m in range(n_modes):
            a, b = Zc[:, :, m].reshape(n, -1), Rc[:, :, m].reshape(n, -1)
            out[f"rev_cos_mode{m}"] = float(R.cosr(b, signs[m] * a).mean()); out[f"rev_cos_raw_mode{m}"] = float(R.cosr(b, a).mean())
            out[f"rev_sign_mode{m}"] = float(signs[m])
        rows_r = {}
        for s, tr, te in R.source_splits(d["GE"], d["SI"]):
            G, _ = R.fit_reader(Z, d["NET"], d["GE"], tr)
            rows_r[d["names"][s]] = dict(n_test=int(te.sum()), r2_rev_net=R.r2(-d["NET"][te], G(d["ZREV"][te])),
                                         cos_rev_net=float(R.cosr(-d["NET"][te] + d["NET"][te].mean(0), G(d["ZREV"][te]) - (-d["NET"][te]).mean(0)).mean()))
        out.update({k: v for k, v in R.aggregate(rows_r).items() if k in ("r2_rev_net", "cos_rev_net")})
    return out


def main(specs):
    code_only = "--code-only" in specs; specs = [x for x in specs if not x.startswith("--")]
    res_path = os.path.join(RESULTS, "stage1_probes.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for spec in specs:
        d = R.load_arrays(spec); tag = os.path.basename(d["path"]).replace("_val.npz", "").replace("swaplam_", "").replace("_readout_arrays.npz", "")
        mf = res.get(tag, {}).get("model_free") if code_only else None
        res[tag] = dict(model_free=mf if mf is not None else model_free(d), code=on_code(d), n_clips=int(len(d["NET"])), legacy=bool(d["legacy"]), path=d["path"])
        json.dump(res, open(res_path, "w"), indent=1)
        mf = res[tag]["model_free"] or {}; c = res[tag]["code"]
        print(f"\n[{tag}] A. frozen dq, per temporal subset (clip-weighted over sources with >= 80 test clips)")
        print("%-12s %7s %6s %9s %9s %9s %9s" % ("subset", "dims", "kept", "r2_net", "r2_seg1", "r2_prof", "r2_seq"))
        for k, v in mf.items():
            print("%-12s %7d %6.3f %9.3f %9.3f %9.3f %9.3f" % (k, v["dims"], v["temporal_var_kept"] or np.nan, v.get("mean_r2_net", np.nan),
                  v.get("mean_r2_seg1", np.nan), v.get("mean_r2_prof", np.nan), v.get("mean_r2_seq", np.nan)))
        print(f"[{tag}] B. code z: dims {c['dims']} r2_net {c.get('mean_r2_net', np.nan):.3f} seg1 {c.get('mean_r2_seg1', np.nan):.3f} "
              f"prof {c.get('mean_r2_prof', np.nan):.3f} seq {c.get('mean_r2_seq', np.nan):.3f} patch_drop {c.get('patch_drop_pct', np.nan):.1f}% "
              + " ".join(f"rev_cos_m{m} {c[f'rev_cos_mode{m}']:+.3f}(sign {c[f'rev_sign_mode{m}']:+.0f})" for m in range(8) if f"rev_cos_mode{m}" in c)
              + (f" r2_rev_net {c['r2_rev_net']:.3f}" if "r2_rev_net" in c else ""), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["rec_src"])
