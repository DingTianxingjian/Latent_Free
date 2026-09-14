"""GPU versions of the exp-1 probes, for feature spaces too wide for sklearn.

The patch-token experiments probe up to 24576-d features (4x4 pooled grid, two
frames concatenated).  sklearn's cholesky Ridge rebuilds the d x d gram for
every alpha, and lbfgs logistic regression on 49k x 24576 is minutes per C.
Both are avoidable: the gram does not depend on alpha, and both problems are
one matmul per step on a GPU.

The objectives are IDENTICAL to probes.py, not merely similar:

  ridge     : features and targets standardized with train statistics, then
              (Xs^T Xs + alpha I) W = Xs^T Ys.  Because standardization makes
              both exactly zero-mean on train, this is what sklearn's
              `Ridge(fit_intercept=True, solver="cholesky")` solves.
  logistic  : 0.5 ||w||^2 + C * sum_i log(1 + exp(-y_i (x_i.w + b))), the
              sklearn LogisticRegression objective, intercept unpenalized.

`python probes_torch.py --verify` refits the real exp-1 CLS probes both ways and
prints the disagreement.  Run it before trusting any number this module makes.
"""

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, r2_score

import data
import probes


def _free():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _to(x, device, dtype=torch.float32):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=device)


def standardize_t(Xtr, Xva):
    """In-place standardization by train statistics.  Returns the train sd."""
    mu = Xtr.mean(dim=0, keepdim=True)
    sd = Xtr.std(dim=0, keepdim=True, unbiased=False)
    sd = torch.where(sd < 1e-8, torch.ones_like(sd), sd)
    Xtr.sub_(mu).div_(sd)
    Xva.sub_(mu).div_(sd)
    return sd.squeeze(0)


def _gram(X, chunk=8192):
    """X^T X accumulated in float64 (X is float32 and can be tall)."""
    d = X.shape[1]
    G = torch.zeros(d, d, dtype=torch.float64, device=X.device)
    for k in range(0, X.shape[0], chunk):
        c = X[k:k + chunk].double()
        G.addmm_(c.T, c)
    return G


def ridge_probe_t(Xtr, Ytr, Xva, Yva, alphas=probes.RIDGE_ALPHAS,
                  device="cuda"):
    """Ridge over the 6 continuous dims.  Mirrors probes.ridge_probe()."""
    Xtr = _to(Xtr, device).clone()
    Xva = _to(Xva, device).clone()
    x_sd = standardize_t(Xtr, Xva)
    return _ridge_std(Xtr, Ytr, Xva, Yva, x_sd, alphas, device)


def _ridge_std(Xtr, Ytr, Xva, Yva, x_sd, alphas, device):
    """Ridge on ALREADY-standardized features.  Consumes Xtr and Xva.

    Split out so a caller that needs both probes can standardize once and hand
    the same tensors to both -- at 24576-d an extra copy is 5 GB.
    """
    Ytr_np = np.asarray(Ytr, dtype=np.float64)
    y_mu = Ytr_np.mean(axis=0, keepdims=True)
    y_sd = Ytr_np.std(axis=0, keepdims=True)
    y_sd = np.where(y_sd < 1e-8, 1.0, y_sd)
    Ys = _to((Ytr_np - y_mu) / y_sd, device, torch.float64)

    d = Xtr.shape[1]
    G = _gram(Xtr)
    B = torch.zeros(d, Ys.shape[1], dtype=torch.float64, device=device)
    for k in range(0, Xtr.shape[0], 8192):
        B.addmm_(Xtr[k:k + 8192].double().T, Ys[k:k + 8192])
    del Xtr, Ys
    Xva = Xva.double()                       # hoisted: d can be 24576
    _free()

    best = None
    for a in alphas:
        Ga = G.clone()
        Ga.diagonal().add_(float(a))         # cheaper than materialising alpha*I
        L, info = torch.linalg.cholesky_ex(Ga)
        del Ga
        if int(info) != 0:                   # too little ridge to stay SPD
            print(f"      [ridge] alpha={a:g} skipped: not positive definite")
            del L
            continue
        W = torch.cholesky_solve(B, L)                       # (d,6) standardized
        del L
        pred = (Xva @ W).cpu().numpy() * y_sd + y_mu
        r2_dim = r2_score(Yva, pred, multioutput="raw_values")
        score = float(np.mean(r2_dim[data.CONT_DIMS]))
        if best is None or score > best["sel_score"]:
            best = dict(sel_score=score, alpha=float(a), r2_dim=r2_dim,
                        W=W.cpu().numpy())
        del W
    assert best is not None, "every alpha failed to factorize"

    # standardized-space -> raw feature space, same convention as probes.py
    W_raw = (y_sd.ravel()[:, None] * best["W"].T) / x_sd.cpu().numpy()[None, :]
    r2_dim = best["r2_dim"]
    out = {f"r2_{g}": float(np.mean(r2_dim[dims]))
           for g, dims in data.DIM_GROUPS.items()}
    out.update({f"r2_dim{k}": float(r2_dim[k]) for k in range(6)})
    out["alpha"] = best["alpha"]
    del G, B, Xva
    _free()
    return out, W_raw


def _fit_logistic(X, y_pm, C, max_iter=200):
    """LBFGS on the sklearn LogisticRegression objective.  y_pm in {-1,+1}."""
    n, d = X.shape
    w = torch.zeros(d, dtype=torch.float32, device=X.device, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, device=X.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=max_iter, history_size=10,
                            tolerance_grad=1e-7, tolerance_change=1e-11,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        m = X @ w + b
        # softplus(-y*m) == log(1 + exp(-y*m)), numerically stable
        loss = 0.5 * (w * w).sum() + C * torch.nn.functional.softplus(-y_pm * m).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def logistic_probe_t(Xtr, ytr, Xva, yva, Cs=probes.LOGIT_CS, device="cuda"):
    Xtr = _to(Xtr, device).clone()
    Xva = _to(Xva, device).clone()
    standardize_t(Xtr, Xva)
    return _logistic_std(Xtr, ytr, Xva, yva, Cs, device)


def _logistic_std(Xtr, ytr, Xva, yva, Cs, device):
    """Gripper probe on ALREADY-standardized features.  Consumes Xtr and Xva."""
    y_pm = _to(np.asarray(ytr) * 2 - 1, device)
    best = None
    for C in Cs:
        w, b = _fit_logistic(Xtr, y_pm, float(C))
        pred = ((Xva @ w + b) > 0).long().cpu().numpy()
        ba = balanced_accuracy_score(yva, pred)
        if best is None or ba > best["bal_acc_grip"]:
            best = dict(bal_acc_grip=float(ba), C=float(C))
    del Xtr, Xva
    _free()
    return best


def run_probe_set_t(Ftr, Fva, ytr_c, yva_c, ytr_g, yva_g, device="cuda",
                    alphas=probes.RIDGE_ALPHAS, Cs=probes.LOGIT_CS,
                    borrow=False):
    """Both probes on one feature matrix, standardizing it exactly once.

    `probes.run_probe_set` standardizes the raw features separately inside each
    probe; doing it once here is the same computation on the same inputs, it
    just does not pay for the copy twice.  With `borrow=True` the caller's
    tensors are standardized in place rather than copied -- worth ~6 GB at
    24576-d, which is the difference between fitting on a 40 GB A100 and not.
    """
    n_train, n_val, dim = len(Ftr), len(Fva), int(Ftr.shape[1])
    Xtr, Xva = _to(Ftr, device), _to(Fva, device)
    # `_to` returns the same object only when no conversion was needed; a fresh
    # tensor is already ours to mutate, so clone exactly when it is not.
    if not borrow and Xtr is Ftr:
        Xtr = Xtr.clone()
    if not borrow and Xva is Fva:
        Xva = Xva.clone()
    x_sd = standardize_t(Xtr, Xva)

    row, W_raw = _ridge_std(Xtr, ytr_c, Xva, yva_c, x_sd, alphas, device)
    # _ridge_std dropped its own references, but these still hold the tensors
    # alive, so the logistic fit reuses the same standardized matrices.
    row.update(_logistic_std(Xtr, ytr_g, Xva, yva_g, Cs, device))
    row["n_train"], row["n_val"], row["dim"] = n_train, n_val, dim
    return row, W_raw


# --------------------------------------------------------------------------- #
def verify(device="cuda"):
    """Refit the real exp-1 CLS probes with sklearn and with this module."""
    index = data.build_index()
    feats = np.load(os.path.join(data.CACHE_DIR, "feats_clean.npy"))
    worst = 0.0
    for stride in data.STRIDES:
        itr, jtr, yc_tr, yg_tr = data.build_pairs(index, stride, "train")
        iva, jva, yc_va, yg_va = data.build_pairs(index, stride, "val")
        for kind in ("diff", "single"):
            Ftr = probes.make_features(feats, itr, jtr, kind)
            Fva = probes.make_features(feats, iva, jva, kind)
            sk, Wsk = probes.run_probe_set(Ftr, Fva, yc_tr, yc_va, yg_tr, yg_va)
            th, Wth = run_probe_set_t(Ftr, Fva, yc_tr, yc_va, yg_tr, yg_va,
                                      device=device)
            keys = ["r2_xyz", "r2_rot", "r2_cont6", "bal_acc_grip"]
            d = {k: abs(sk[k] - th[k]) for k in keys}
            wcos = float(np.abs((Wsk * Wth).sum(1) /
                                (np.linalg.norm(Wsk, axis=1) *
                                 np.linalg.norm(Wth, axis=1) + 1e-12)).min())
            worst = max(worst, max(d.values()))
            print(f"stride {stride} {kind:6s} "
                  + "  ".join(f"{k}: sk {sk[k]:.4f} / torch {th[k]:.4f} "
                              f"(d {d[k]:.1e})" for k in keys)
                  + f"   alpha sk {sk['alpha']:g} torch {th['alpha']:g}"
                  + f"   min |cos| between ridge weight rows {wcos:.6f}")
    print(f"\n[verify] worst metric disagreement: {worst:.2e}")
    assert worst < 5e-3, "torch probes disagree with sklearn -- do not use"
    print("[verify] OK -- torch probes reproduce the sklearn protocol")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.verify:
        verify(a.device)
