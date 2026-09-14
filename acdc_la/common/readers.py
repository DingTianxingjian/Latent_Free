"""Offline readers over a validation dump: the per-source episode-disjoint split, ridge fitting,
clip-weighted aggregation, and the loader that also understands the legacy
`claude_survey/results/swaplam_<tag>_readout_arrays.npz` files (so every ledger number can be
recomputed here).

Conventions (identical to the ledger's mixprobes):  per source, 1/3 of the EPISODES are test
(RandomState(1)); alpha of the ridge is picked on an inner episode split; aggregates are
clip-weighted over sources with >= 80 test clips; `umean_*` is the unweighted mean.
"""
import os
import numpy as np

from ..paths import RESULTS, LEGACY_RESULTS
from .probes import _ridge, _r2, standardise

MIN_CLIPS, TEST_FRAC = int(os.environ.get("ACDC_MIN_CLIPS", 80)), 1 / 3   # ACDC_MIN_CLIPS=10 for smoke dumps


def load_arrays(spec):
    """spec: path, this repo's tag (results/<tag>_val.npz) or legacy tag (swaplam_<tag>_readout_arrays.npz).
    Returns a dict with at least Z, DF, NET, SEG1, GE, SI, names and (if present) the rest, normalised:
    NET (n,3) net xyz; SEG1 (n,3); PROF (n,K*3) mean-removed sequence; SEQ (n,K*3); ASW_NET donor net xyz."""
    cands = [spec, os.path.join(RESULTS, f"{spec}_val.npz"), os.path.join(LEGACY_RESULTS, f"swaplam_{spec}_readout_arrays.npz")]
    path = next((p for p in cands if os.path.exists(p)), None)
    assert path, f"no dump for {spec}"
    z = np.load(path, allow_pickle=True); d = {k: z[k] for k in z.files}
    d["legacy"] = "ACT" not in d
    if not d["legacy"]:
        A = d["ACT"].astype(np.float64); d["NET"], d["SEG1"] = A.sum(1), A[:, 0]
        d["SEQ"] = A.reshape(len(A), -1); d["PROF"] = (A - A.mean(1, keepdims=True)).reshape(len(A), -1)
        if "ASW" in d: d["ASW_NET"] = d["ASW"].astype(np.float64).sum(1)
    else:
        d["NET"], d["SEG1"] = d["NET"].astype(np.float64), d["SEG1"].astype(np.float64)
        if "ASW" in d: d["ASW_NET"] = d["ASW"].astype(np.float64)
    d["names"] = [str(x) for x in d["names"]]
    d["path"] = path
    return d


def source_splits(GE, SI, test_frac=TEST_FRAC, seed=1, min_train=20, min_test=10):
    for s in np.unique(SI):
        m = SI == s; ug = np.unique(GE[m]); rs = np.random.RandomState(seed)
        te_eps = rs.choice(ug, max(1, int(len(ug) * test_frac)), replace=False)
        te = m & np.isin(GE, te_eps); tr = m & ~np.isin(GE, te_eps)
        if tr.sum() < min_train or te.sum() < min_test: continue
        yield int(s), tr, te


def fit_reader(X, Y, GE, tr):
    """Ridge X -> Y on rows `tr`; returns a callable G(X) and the chosen alpha."""
    mu, sd, W, ym, al = _ridge(np.asarray(X[tr], np.float64), np.asarray(Y[tr], np.float64), GE[tr])
    return (lambda Xo: standardise(Xo, mu, sd) @ W + ym), al


def cosr(A, B):
    return (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9)


def r2(Y, P): return _r2(np.asarray(Y, np.float64), np.asarray(P, np.float64))


def aggregate(rows, min_clips=None, n_key="n_test"):
    min_clips = MIN_CLIPS if min_clips is None else min_clips
    """rows: {source: {metric: value, n_test: int}} -> {metric: clip-weighted mean, umean_metric, n_sources, per_source}."""
    out = {}
    keys = sorted({k for r in rows.values() for k in r if k != n_key and isinstance(r[k], (int, float, np.floating))})
    for k in keys:
        v = np.array([(r[k], r[n_key]) for r in rows.values() if k in r and r[n_key] >= min_clips and np.isfinite(r[k])], np.float64)
        if len(v): out[k] = float((v[:, 0] * v[:, 1]).sum() / v[:, 1].sum()); out[f"umean_{k}"] = float(v[:, 0].mean())
    out["n_sources"] = int(sum(r[n_key] >= min_clips for r in rows.values())); out["per_source"] = rows
    return out


def table(res, cols, tags, title="metric"):
    print("%-26s" % title + "".join("%12s" % t[:11] for t in tags))
    for k in cols:
        print("%-26s" % k + "".join("%12.3f" % res[t].get(k, float("nan")) for t in tags))
