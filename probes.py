"""Linear probes + metrics shared by exp1.py and exp2.py.

Protocol (identical everywhere, so numbers are comparable across conditions):
  * features and continuous targets are standardized with TRAIN statistics only;
  * ridge for the 6 continuous dims, logistic regression for the binary gripper;
  * the regularisation strength is picked on val (see caveat in SUMMARY.md);
  * R^2 is reported in the original action units, uniform-averaged within a
    dimension group; the gripper uses balanced accuracy.
"""

import os

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score

import data


def _grid(env, default):
    return tuple(float(x) for x in os.environ.get(env, default).split(","))


# Defaults are the grids specified in the brief.  They can be widened via
# LF_RIDGE_ALPHAS / LF_LOGIT_CS -- needed because the selected alpha lands on the
# top of the specified grid, and a boundary-pinned hyperparameter means the grid,
# not the data, is setting the answer.
RIDGE_ALPHAS = _grid("LF_RIDGE_ALPHAS", "0.01,0.1,1,10,100")
LOGIT_CS = _grid("LF_LOGIT_CS", "0.01,0.1,1,10,100")
OUT_SUFFIX = os.environ.get("LF_OUT_SUFFIX", "")


def standardize(Xtr, Xva):
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (Xtr - mu) / sd, (Xva - mu) / sd, mu.ravel(), sd.ravel()


def ridge_probe(Xtr, Ytr, Xva, Yva, alphas=RIDGE_ALPHAS):
    """Ridge on the 6 continuous dims.  Returns metrics + raw-space weights.

    The returned `W_raw` maps *unstandardized* features to actions; its 6 rows
    span the "action subspace" used by the exp-2 PCA analysis.
    """
    Xtr_s, Xva_s, _, x_sd = standardize(Xtr, Xva)
    y_mu = Ytr.mean(axis=0, keepdims=True)
    y_sd = np.where(Ytr.std(axis=0, keepdims=True) < 1e-8, 1.0,
                    Ytr.std(axis=0, keepdims=True))
    Ytr_s = (Ytr - y_mu) / y_sd

    best = None
    for a in alphas:
        m = Ridge(alpha=a, fit_intercept=True, solver="cholesky")
        m.fit(Xtr_s, Ytr_s)
        pred = m.predict(Xva_s) * y_sd + y_mu
        r2_dim = r2_score(Yva, pred, multioutput="raw_values")
        score = float(np.mean(r2_dim[data.CONT_DIMS]))
        if best is None or score > best["sel_score"]:
            best = dict(sel_score=score, alpha=a, r2_dim=r2_dim, model=m)

    m = best["model"]
    # standardized-space coefs -> raw feature space (span is what matters)
    W_raw = (y_sd.ravel()[:, None] * m.coef_) / x_sd[None, :]
    r2_dim = best["r2_dim"]
    out = {f"r2_{g}": float(np.mean(r2_dim[dims]))
           for g, dims in data.DIM_GROUPS.items()}
    out.update({f"r2_dim{k}": float(r2_dim[k]) for k in range(6)})
    out["alpha"] = best["alpha"]
    return out, W_raw


def logistic_probe(Xtr, ytr, Xva, yva, Cs=LOGIT_CS):
    """Gripper probe.  Returns balanced accuracy at the best C."""
    Xtr_s, Xva_s, _, _ = standardize(Xtr, Xva)
    best = None
    for C in Cs:
        m = LogisticRegression(C=C, max_iter=500, n_jobs=-1)
        m.fit(Xtr_s, ytr)
        ba = balanced_accuracy_score(yva, m.predict(Xva_s))
        if best is None or ba > best["bal_acc_grip"]:
            best = dict(bal_acc_grip=float(ba), C=C)
    return best


def run_probe_set(Ftr, Fva, ytr_c, yva_c, ytr_g, yva_g):
    """Ridge + logistic on one feature matrix.  Returns (row dict, W_raw)."""
    row, W_raw = ridge_probe(Ftr, ytr_c, Fva, yva_c)
    row.update(logistic_probe(Ftr, ytr_g, Fva, yva_g))
    row["n_train"] = len(Ftr)
    row["n_val"] = len(Fva)
    return row, W_raw


def make_features(feats, i, j, kind):
    """`diff` = f(o_{t+s}) - f(o_t);  `single` = f(o_t) alone."""
    if kind == "diff":
        return feats[j] - feats[i]
    if kind == "single":
        return feats[i]
    raise ValueError(kind)
