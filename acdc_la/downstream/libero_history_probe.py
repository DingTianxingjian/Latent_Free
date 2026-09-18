"""How much of the latent a VLA must predict can the deployment input determine — and does the OBSERVED past help?

The VLA sees one frame + the instruction and must produce z(t), the code of the window [t, t+12] (5 frames, stride 3,
exactly the label `downstream/vla/datasets_ctrl.py` trains on).  z(t) is computed from future frames, so part of it is
not determined by the input.  The past window [t-12, t] HAS been observed at deployment, and the same tokenizer can
encode it without prediction: z_past(t) = code of frames t-12 ... t.  This probe measures, on held-out LIBERO episodes,
how much of z(t) (and of the action chunk) each input determines:

  state        : PCA of the current frame's DINO grid + task one-hot     (what the VLA has now)
  state+zpast  : + z_past(t) from the observed past frames                (the proposed history input)
  zpast        : z_past(t) alone
  state+dpast  : + PCA of the raw past displacement F(t) - F(t-12)       (reference: history without the tokenizer)

Ridge per target block, alpha chosen on an inner episode split, R2 = 1 - SSE/SST per dim averaged over dims, on the
index's validation episodes.  Only windows with 12 steps of past AND future inside the episode are used, so every
condition sees the same rows.

  python -m acdc_la.downstream.libero_history_probe --tag latp2_inv
"""
import os, json, time, argparse
import numpy as np
import torch

from ..paths import CKPT, RESULTS
from ..common.model import load_ckpt
from . import libero_offline as L

W, STRIDE, T = 12, 3, 5                       # the VLA label: window of 12 steps, 5 frames at stride 3


@torch.no_grad()
def codes(tok, dstats, G, starts, bs=256):
    out = []
    for i in range(0, len(starts), bs):
        t = torch.from_numpy(starts[i:i + bs]); rows = t[:, None] + torch.arange(0, W + 1, STRIDE)[None]
        g4 = G[rows].float(); dF = (g4[:, 1:] - g4[:, :-1]) / dstats
        out.append(tok.quantize(g4[:, 0], dF)[0].flatten(1))
    return torch.cat(out).numpy().astype(np.float64)


def pca_fit(X, k):
    Xt = torch.from_numpy(X).float(); mu = Xt.mean(0)
    _, _, V = torch.pca_lowrank(Xt - mu, q=k, center=False, niter=4)            # randomised: 12288-d frames
    return mu.numpy(), V.numpy()


def ridge_r2(Xtr, Ytr, Xte, Yte, inner, alphas=(1e-1, 1, 10, 100, 1e3, 1e4)):
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr = np.clip((Xtr - mx) / sx, -10, 10); Xte = np.clip((Xte - mx) / sx, -10, 10)
    my = Ytr.mean(0); Ytr = Ytr - my; Yte = Yte - my
    def fit(X, Y, a): return np.linalg.solve(X.T @ X + a * np.eye(X.shape[1]), X.T @ Y)
    def r2(Y, P):
        sst = ((Y - Y.mean(0)) ** 2).sum(0); keep = sst > 1e-9
        return float((1 - ((Y - P) ** 2).sum(0)[keep] / sst[keep]).mean())
    a_best = max(alphas, key=lambda a: r2(Ytr[inner], Xtr[inner] @ fit(Xtr[~inner], Ytr[~inner], a)))
    return r2(Yte, Xte @ fit(Xtr, Ytr, a_best)), a_best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="latp2_inv"); ap.add_argument("--pca", type=int, default=256)
    a = ap.parse_args(); t0 = time.time(); torch.set_num_threads(16)
    G, idx = L.frame_features(None, "cpu")
    tok, dstats, _ = load_ckpt(os.path.join(CKPT, f"{a.tag}.pt"), "cpu"); tok.eval()
    zs = np.load(os.path.join(CKPT, f"z_stats_{a.tag}.npz"))
    rows, eps = [], []
    for e, (s, n) in enumerate(zip(idx["starts"], idx["lengths"])):
        for t in range(int(s) + W, int(s + n) - W): rows.append(t); eps.append(e)
    rows, eps = np.array(rows), np.array(eps)
    is_val = idx["is_val"].astype(bool)[eps]; task = idx["task_id"][eps].astype(np.int64)
    print(f"[hist] {a.tag}: {len(rows)} windows, {len(np.unique(eps))} episodes, {int(is_val.sum())} held-out windows, "
          f"{len(np.unique(task))} tasks ({time.time()-t0:.0f}s)", flush=True)
    Zf = (codes(tok, dstats, G, rows) - zs["mean"]) / zs["std"]           # the VLA's standardised target
    Zp = (codes(tok, dstats, G, rows - W) - zs["mean"]) / zs["std"]       # observed past window, same tokenizer
    A = idx["actions"]; chunk = np.stack([A[t:t + W].reshape(-1) for t in rows])
    Fnow = G[torch.from_numpy(rows)].float().flatten(1).numpy()
    Dpast = Fnow - G[torch.from_numpy(rows - W)].float().flatten(1).numpy()
    tr, te = ~is_val, is_val
    mu, P = pca_fit(Fnow[tr], a.pca); S_ = ((Fnow - mu) @ P).astype(np.float64); del Fnow
    mu2, P2 = pca_fit(Dpast[tr], a.pca); D_ = ((Dpast - mu2) @ P2).astype(np.float64); del Dpast
    onehot = np.eye(task.max() + 1)[task]
    tr_eps = np.unique(eps[tr]); rng = np.random.RandomState(0); inner_eps = rng.choice(tr_eps, len(tr_eps) // 5, replace=False)
    inner = np.isin(eps[tr], inner_eps)
    conds = {"state": np.hstack([S_, onehot]), "state+zpast": np.hstack([S_, onehot, Zp]), "zpast": Zp,
             "state+dpast": np.hstack([S_, onehot, D_])}
    R = {"tag": a.tag, "n_windows": int(len(rows)), "n_test": int(te.sum()), "window": W, "stride": STRIDE, "pca": a.pca}
    print(f"{'input':14s} {'R2 z(t)':>9s} {'R2 action chunk':>16s}", flush=True)
    for c, X in conds.items():
        rz, az = ridge_r2(X[tr], Zf[tr], X[te], Zf[te], inner)
        ra, aa = ridge_r2(X[tr], chunk[tr], X[te], chunk[te], inner)
        R[c] = dict(r2_z=rz, r2_action=ra, alpha_z=az, alpha_action=aa)
        print(f"{c:14s} {rz:9.3f} {ra:16.3f}   (alpha {az:g} / {aa:g}, dims {X.shape[1]})", flush=True)
    json.dump(R, open(os.path.join(RESULTS, f"history_probe_{a.tag}.json"), "w"), indent=1)
    print(f"[hist] wrote results/history_probe_{a.tag}.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
