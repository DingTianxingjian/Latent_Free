"""Per-source held-out probes for the OXE mixture.

WHY PER SOURCE.  The first pooled read (eval_mix job 20444100) gave NMSE > 1 for every model
AND for the frozen-feature reference (1.117, vs ~0.64 on Bridge alone).  The mixture's action
spaces are heterogeneous -- Bridge xyz in metres (~0.03 per segment), ucsd_kitchen in
millimetres (hundreds), GNM waypoint offsets, absolute-pose differences -- so one ridge over
the pooled set is dominated by whichever source has the largest units and predicts nothing
for the rest.  A probe is fit and read WITHIN each source (episode-disjoint inside the
source), and the report carries the per-source table plus a clip-weighted and an unweighted
mean.  Sources with fewer than `min_clips` held-out clips are listed but not averaged.

The same fitted probe (per source) is reused for the moving-patch view (drop) and for the
re-encoded swapped transition (transport_action_r2), so those numbers are read on the same
coordinates as the clean number.
"""
import os
import numpy as np


def _r2(Y, P):
    """R2 averaged over the target dims that VARY on these rows.  A constant dim (z of the 2-D GNM navigation
    sources is exactly 0; a per-episode nuisance target on a single test episode) is excluded instead of
    contributing a meaningless 1.0 (legacy +1e-9 convention with an exact-zero prediction) or -1e13; nan
    when no dim varies."""
    var = ((Y - Y.mean(0)) ** 2).sum(0); ok = var > 1e-9 * max(len(Y), 1) * (np.abs(Y).mean() ** 2 + 1e-12)
    if not ok.any(): return float("nan")
    return float(np.mean(1 - ((Y - P) ** 2).sum(0)[ok] / var[ok]))


CLIP = 10.0   # standardised features are clipped to +-CLIP at fit AND read time: a rare quantised code that is
              # 50 sd from the training mean must not turn a linear reader into a 1e6 prediction


def standardise(X, mu, sd):
    return np.clip((np.asarray(X, np.float64) - mu) / sd, -CLIP, CLIP)


def _ridge(Xtr, Ytr, Gtr, alphas=(1e-1, 1e0, 1e1, 1e2, 1e3, 1e4)):
    """Episode-disjoint inner split picks alpha.  Returns (mu, sd, W, ym, alpha).
    Dual form when d > n (the 12,288-d frozen-feature reference): the Gram matrix is computed ONCE and
    sliced for the inner split, so a 7-fit alpha search costs one n x d x n product, not seven."""
    Xtr, Ytr = np.asarray(Xtr, np.float64), np.asarray(Ytr, np.float64)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    # a column constant on the training rows carries nothing and, divided by ~0, would blow up on any test row
    # where it is not constant (quantised UniVLA codes: 5/16 codebook entries used); drop it by making it 0 everywhere
    sd = np.where(sd < 1e-3 * max(sd.max(), 1e-12), np.inf, sd + 1e-6); Xn = standardise(Xtr, mu, sd); ym = Ytr.mean(0); Yc = Ytr - ym
    ug = np.unique(Gtr); rs = np.random.RandomState(0)
    hold = np.isin(Gtr, rs.choice(ug, max(1, len(ug) // 5), replace=False))
    if hold.all() or (~hold).all(): hold = np.zeros(len(Gtr), bool); hold[: len(Gtr) // 5] = True
    n, d = Xn.shape
    if d > n:
        K = Xn @ Xn.T; f, h = ~hold, hold
        def pred_hold(al):                  # dual coefficients on the fit rows, predictions on the held rows
            a = np.linalg.solve(K[np.ix_(f, f)] + al * np.eye(f.sum()), Yc[f]); return K[np.ix_(h, f)] @ a + ym
        best = max(((_r2(Ytr[h], pred_hold(al)), al) for al in alphas))
        W = Xn.T @ np.linalg.solve(K + best[1] * np.eye(n), Yc)
    else:
        def fit(X, Y, al): return np.linalg.solve(X.T @ X + al * np.eye(X.shape[1]), X.T @ (Y - ym))
        best = max(((_r2(Ytr[hold], Xn[hold] @ fit(Xn[~hold], Ytr[~hold], al) + ym), al) for al in alphas))
        W = fit(Xn, Ytr, best[1])
    return mu, sd, W, ym, best[1]


def per_source_readouts(Z, targets, GE, SI, names, extra=None, min_clips=None, test_frac=1 / 3):
    min_clips = int(os.environ.get("ACDC_MIN_CLIPS", 80)) if min_clips is None else min_clips
    """Z (n,d); targets: {name: (n,k)} -- each gets its own probe; extra: {label: (Zother (n,d),
    target_name, Yother or None)} -- read with the probe fitted for `target_name` on the clean Z
    (Yother defaults to that target; e.g. transport uses the donor's action).
    Returns a flat dict: per-source `<src>/<metric>`, plus `mean_<metric>` (clip-weighted over
    sources with >= min_clips test clips) and `umean_<metric>` (unweighted)."""
    Z = np.asarray(Z, np.float64); out, table = {}, {}
    for s in np.unique(SI):
        m = SI == s; ug = np.unique(GE[m]); rs = np.random.RandomState(1)
        te_eps = rs.choice(ug, max(1, int(len(ug) * test_frac)), replace=False)
        te = m & np.isin(GE, te_eps); tr = m & ~np.isin(GE, te_eps)
        row = dict(n_train=int(tr.sum()), n_test=int(te.sum()), n_eps=int(len(ug)))
        if tr.sum() < 20 or te.sum() < 10:
            table[names[s]] = row; continue
        probes = {}
        for tname, Y in targets.items():
            Y = np.asarray(Y, np.float64)
            mu, sd, W, ym, al = _ridge(Z[tr], Y[tr], GE[tr]); probes[tname] = (mu, sd, W, ym)
            row[f"r2_{tname}"] = _r2(Y[te], standardise(Z[te], mu, sd) @ W + ym); row[f"alpha_{tname}"] = al
        for label, spec in (extra or {}).items():
            Zo, tname, Yo = spec[:3]; mask = spec[3] if len(spec) > 3 else None
            mu, sd, W, ym = probes[tname]; Y = np.asarray(targets[tname] if Yo is None else Yo, np.float64)
            sel = te if mask is None else (te & np.asarray(mask, bool))      # e.g. transport: same-source donors only
            if sel.sum() < 10: continue
            Pd = standardise(np.asarray(Zo, np.float64)[sel], mu, sd) @ W + ym
            row[f"r2_{label}"] = _r2(Y[sel], Pd); row[f"n_{label}"] = int(sel.sum())
            # scale-free: cosine between predicted and true (mean-removed) vectors, so an exaggerated but
            # directionally right decode is not rewarded or punished for its scale
            Yc, Pc = Y[sel] - Y[sel].mean(0), Pd - Y[sel].mean(0)
            row[f"cos_{label}"] = float(np.mean((Yc * Pc).sum(1) / (np.linalg.norm(Yc, axis=1) * np.linalg.norm(Pc, axis=1) + 1e-9)))
        table[names[s]] = row
    metrics = sorted({k for r in table.values() for k in r if k.startswith(("r2_", "cos_"))})
    for met in metrics:
        rows = [(r[met], r["n_test"]) for r in table.values() if met in r and r["n_test"] >= min_clips and np.isfinite(r[met])]
        if rows:
            w = np.array([n for _, n in rows], float); v = np.array([x for x, _ in rows])
            out[f"mean_{met}"] = float((v * w).sum() / w.sum()); out[f"umean_{met}"] = float(v.mean()); out[f"n_sources_{met}"] = len(rows)
    out["per_source"] = table
    return out
