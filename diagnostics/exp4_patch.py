"""Experiment 4 -- is the Exp-1 result about frame differences, or about CLS?

Exp 1 probed the CLS token: one 768-d global summary per frame.  Exp 3 then
found UniVLA's stage-1 LAM, which reads the full patch-token grid of two frames,
decodes actions far better (pre-VQ xyz R^2 0.80 at stride 4) than CLS
differences do (0.55).  Two different things could produce that gap:

  (a) the CLS bottleneck destroys the action signal, and patch tokens keep it;
  (b) the LAM is not reading the transition at all, but reading the STATE out of
      the two frames -- arm pose is visible in either frame alone, and in
      scripted demos the action at time t is highly predictable from the pose at
      time t.  That is exactly the confound the Exp-1 `single` ablation exposed,
      and a two-frame encoder is free to exploit it.

Neither Exp 1 nor Exp 3 can separate these, because there is no single-frame
baseline in patch space.  This experiment supplies it, as a full matrix:

  feature space   cls    768-d CLS token                       (what exp 1 used)
                  pmean  768-d mean over the 256 patch tokens  (dim-matched to
                         cls, so "cls vs pmean" isolates the pooling)
                  pg4    12288-d 4x4 average-pooled token grid (keeps coarse
                         spatial layout; closest to what the LAM sees)

  input           single     f(o_t)                  -- state only, no transition
                  diff       f(o_{t+s}) - f(o_t)     -- what a LAM is claimed to
                                                        compress
                  pair       [f(o_t), f(o_{t+s})]    -- everything a two-frame
                                                        encoder can linearly see
                  pair_shuf  [f(o_t), f(o_{t'})]     -- CONTROL: t' uniform in the
                                                        same episode, target
                                                        unchanged

`pair` is the matched control for a LAM: same inputs, linear readout.  As a
function class it contains both `single` and `diff` (`diff` is `pair` with the
weights constrained antisymmetric), so it upper-bounds them.  The decomposition
that matters is therefore

    single        = how much is just state correlation
    pair - single = how much the second frame genuinely adds

and a LAM latent should be compared against `single` in the SAME space before
any of its score is credited to transition information.

`pair` is 2x the width of `single`/`diff`, so part of any gain could be probe
capacity rather than information.  `pair_shuf` removes that ambiguity: it has
exactly the same width and the same "one frame is o_t" structure, but its second
frame carries no information about which transition happened.  Whatever `pair`
scores above `pair_shuf` is specific to the actual next frame.  (`cls` vs
`pmean` is capacity-matched by construction and needs no such control.)
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

import data
import features_patch
import probes
import probes_torch
import viz

SPACES = ("cls", "pmean", "pg4")
SPACE_LABEL = {"cls": "CLS token\n(768-d)",
               "pmean": "patch mean\n(768-d)",
               "pg4": "patch 4x4 grid\n(12288-d)",
               "pg4pca768": "patch 4x4 grid\nPCA to 768-d",
               "pg4pca512": "patch 4x4 grid\nPCA to 512-d"}
KINDS = ("single", "diff", "pair", "pair_shuf")
KIND_LABEL = {"single": r"$f(o_t)$  single frame",
              "diff": r"$\Delta f$  difference",
              "pair": r"$[f(o_t), f(o_{t+s})]$  both frames",
              "pair_shuf": r"$[f(o_t), f(o_{t'})]$  control, $t'$ random"}
SHUF_SEED = 123
# Plot ordering, narrowest first, so the width control reads left-to-right.
PLOT_ORDER = ("cls", "pmean", "pg4pca512", "pg4pca768", "pg4")

# Wider than the brief's grid in both directions: exp 1 selected the top of the
# grid (alpha=100) and exp 3 the bottom (alpha=0.01), and these feature spaces
# span 768 to 24576 dims, so the right amount of shrinkage is not comparable
# across rows.  The gram is reused across alphas, so this costs almost nothing.
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5)


def load_space(space, condition, device="cuda"):
    if space == "cls":
        return np.load(os.path.join(data.CACHE_DIR, f"feats_{condition}.npy"))
    if space.startswith("pg4pca"):
        return pca_reduce(condition, int(space[len("pg4pca"):]), device)
    pool = {"pmean": "mean", "pg4": "g4"}[space]
    return np.load(features_patch.pool_path(condition, pool))


def pca_reduce(condition, k, device="cuda"):
    """Top-k PCA of the 12288-d patch grid, fitted on TRAIN FRAMES ONLY.

    `pg4` beats `cls` at single-frame decoding, but it is also 16x wider, so
    that gap confounds "spatial layout carries the signal" with "a wider probe
    fits better".  Projecting the same features down to 768-d (matched to cls
    and pmean) and to 512-d (matched to the UniVLA latent) separates the two:
    whatever survives the projection is information, not width.

    Fitting on train frames only matters -- the eigenbasis is estimated from
    data, so fitting it on val frames would leak them into the probe.
    """
    cache = os.path.join(data.CACHE_DIR, f"pfeats_{condition}_g4pca{k}.npy")
    if os.path.exists(cache):
        return np.load(cache)

    index = data.build_index()
    X = torch.from_numpy(np.load(features_patch.pool_path(condition, "g4"))).to(device)
    tr = np.concatenate([np.arange(int(index["starts"][e]),
                                   int(index["starts"][e]) + int(index["lengths"][e]))
                         for e in range(len(index["starts"]))
                         if not bool(index["is_val"][e])])
    print(f"[pca] fitting {k}-d basis on {len(tr)} train frames "
          f"of {X.shape[0]} ({X.shape[1]}-d)", flush=True)
    Xt = X[torch.from_numpy(tr).to(device)]
    mu = Xt.mean(dim=0, keepdim=True)
    C = torch.zeros(X.shape[1], X.shape[1], dtype=torch.float64, device=device)
    for i in range(0, Xt.shape[0], 8192):
        c = (Xt[i:i + 8192] - mu).double()
        C.addmm_(c.T, c)
    del Xt
    evals, evecs = torch.linalg.eigh(C / len(tr))       # ascending
    V = evecs[:, -k:].flip(-1).contiguous()             # top-k, descending
    var = float(evals[-k:].sum() / evals.clamp(min=0).sum())
    print(f"[pca] top-{k} keeps {var:.1%} of train variance", flush=True)
    del C, evals, evecs

    out = np.empty((X.shape[0], k), dtype=np.float32)
    for i in range(0, X.shape[0], 8192):
        out[i:i + 8192] = ((X[i:i + 8192] - mu).double() @ V).float().cpu().numpy()
    del X, V
    torch.cuda.empty_cache()
    np.save(cache, out)
    print(f"[pca] saved {cache} {out.shape}", flush=True)
    return out


def make_features_t(F, i, j, kind, device=None):
    """Build one probe matrix by gathering rows of the cached feature table.

    `F` may live on the GPU (fastest) or in host memory (`--feats-cpu`, which
    trades a per-cell transfer for ~3 GB of device memory -- enough to make the
    widest cell fit on a 40 GB A100).
    """
    if kind == "single":
        X = F[i]
    elif kind == "diff":
        X = F[j] - F[i]
    elif kind in ("pair", "pair_shuf"):         # j is already the right partner
        X = torch.cat([F[i], F[j]], dim=1)
    else:
        raise ValueError(kind)
    return X if device is None or X.device.type == device else X.to(device)


def shuffled_partners(index, stride, split, n_expect, seed=SHUF_SEED):
    """Global row ids of a random frame from the SAME episode, one per pair.

    Episodes are visited in exactly the order data.build_pairs() uses, so entry
    k here is the partner for pair k there.  The target is left untouched: the
    probe is still asked for the action over [t, t+stride), it just no longer
    gets to see where the arm went.
    """
    rng = np.random.RandomState(seed + 7 * stride + (1 if split == "val" else 0))
    out = []
    for e in range(len(index["starts"])):
        if bool(index["is_val"][e]) != (split == "val"):
            continue
        s, T = int(index["starts"][e]), int(index["lengths"][e])
        n = T - stride
        if n <= 0:
            continue
        out.append(s + rng.randint(0, T, size=n))
    js = np.concatenate(out).astype(np.int64)
    assert len(js) == n_expect, (len(js), n_expect)
    return js


def run(condition="clean", spaces=SPACES, kinds=KINDS, strides=data.STRIDES,
        device="cuda", out_csv=None, budget_min=None, feats_cpu=False):
    """Probe every (space, stride, kind) cell.

    Results are flushed to `out_csv` after every cell and completed cells are
    skipped on a rerun, so this survives a short wall clock: resubmit until the
    run reports nothing left.  Cheap spaces come first, so a truncated run still
    answers the capacity-matched `cls` vs `pmean` question.
    """
    index = data.build_index()
    pairs, shuf = {}, {}
    for s in strides:
        pairs[s] = (data.build_pairs(index, s, "train"),
                    data.build_pairs(index, s, "val"))
        shuf[s] = (shuffled_partners(index, s, "train", len(pairs[s][0][0])),
                   shuffled_partners(index, s, "val", len(pairs[s][1][0])))

    rows, done = [], set()
    if out_csv and os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        prev = prev[prev.condition == condition]
        rows = prev.to_dict("records")
        done = set(map(tuple, prev[["space", "stride", "feature"]]
                       .drop_duplicates().values))
        print(f"[exp4] resuming: {len(done)} cells already in {out_csv}")

    t0, took, pending = time.time(), [], []
    for space in spaces:
        if all((space, s, k) in done for s in strides for k in kinds):
            print(f"[exp4] space {space}: all cells cached, skipping")
            continue
        feats = load_space(space, condition, device)
        fdev = "cpu" if feats_cpu else device
        F = torch.from_numpy(feats).to(fdev)
        print(f"\n[exp4] space {space}: {tuple(feats.shape)} "
              f"({F.element_size() * F.nelement() / 2**30:.2f} GiB on {fdev})",
              flush=True)
        for stride in strides:
            (itr, jtr, yc_tr, yg_tr), (iva, jva, yc_va, yg_va) = pairs[stride]
            it = torch.from_numpy(itr).to(fdev)
            jt = torch.from_numpy(jtr).to(fdev)
            iv = torch.from_numpy(iva).to(fdev)
            jv = torch.from_numpy(jva).to(fdev)
            jts = torch.from_numpy(shuf[stride][0]).to(fdev)
            jvs = torch.from_numpy(shuf[stride][1]).to(fdev)
            for kind in kinds:
                if (space, stride, kind) in done:
                    continue
                if budget_min is not None and took:
                    left = budget_min * 60 - (time.time() - t0)
                    est = max(took) * 1.2
                    if est > left:
                        print(f"   s={stride} {space} {kind}: DEFERRED, needs "
                              f"~{est / 60:.1f} min, {left / 60:.1f} min left",
                              flush=True)
                        pending.append((space, stride, kind))
                        continue
                t1 = time.time()
                sh = kind == "pair_shuf"
                # `borrow`: these matrices are rebuilt per cell and are up to
                # 5 GB each, so let the probes standardize them in place.
                Ftr = make_features_t(F, it, jts if sh else jt, kind, device)
                Fva = make_features_t(F, iv, jvs if sh else jv, kind, device)
                res, W = probes_torch.run_probe_set_t(
                    Ftr, Fva, yc_tr, yc_va, yg_tr, yg_va, device=device,
                    alphas=ALPHAS, borrow=True)
                del Ftr, Fva
                torch.cuda.empty_cache()

                took.append(time.time() - t1)
                pin = ("  <-- alpha at grid edge"
                       if res["alpha"] in (ALPHAS[0], ALPHAS[-1]) else "")
                print(f"   s={stride} {space:6s} {kind:9s} d={res['dim']:<6d} "
                      f"alpha={res['alpha']:<8g} C={res['C']:<6g} "
                      f"xyz={res['r2_xyz']:.3f} rot={res['r2_rot']:.3f} "
                      f"cont6={res['r2_cont6']:.3f} "
                      f"grip={res['bal_acc_grip']:.3f}  "
                      f"[{took[-1]:.0f}s]{pin}", flush=True)

                base = dict(condition=condition, stride=stride, space=space,
                            feature=kind, dim=res["dim"],
                            n_train=res["n_train"], n_val=res["n_val"])
                for g in ("xyz", "rot", "cont6"):
                    rows.append(dict(base, dim_group=g, metric="r2",
                                     value=res[f"r2_{g}"],
                                     hyperparam=f"alpha={res['alpha']:g}"))
                rows.append(dict(base, dim_group="gripper",
                                 metric="balanced_accuracy",
                                 value=res["bal_acc_grip"],
                                 hyperparam=f"C={res['C']:g}"))
                for k in range(6):
                    rows.append(dict(base,
                                     dim_group=["dx", "dy", "dz",
                                                "drx", "dry", "drz"][k],
                                     metric="r2", value=res[f"r2_dim{k}"],
                                     hyperparam=f"alpha={res['alpha']:g}"))
                if out_csv:                     # flush after every cell
                    pd.DataFrame(rows).to_csv(out_csv, index=False)
            del it, jt, iv, jv, jts, jvs
        del F
        torch.cuda.empty_cache()

    if pending:
        print(f"\n[exp4] {len(pending)} cells DEFERRED for lack of wall time: "
              f"{pending}\n[exp4] resubmit to continue -- finished cells are "
              f"cached in {out_csv}")
    else:
        print(f"\n[exp4] all cells done in {(time.time() - t0) / 60:.1f} min")
    return pd.DataFrame(rows), pending


def decomposition(df):
    """Split each two-frame score into state / capacity / transition.

      single         what o_t alone already gives away (pure state correlation)
      pair_shuf      same width as `pair`, second frame uninformative
      capacity_gain  pair_shuf - single: what the extra width buys for free
      transition     pair - pair_shuf: the part that needs the ACTUAL next frame
      frac_state     single / pair: how much of the two-frame score was already
                     there without any transition information at all
    """
    out = []
    for (stride, space, g), sub in df[df.metric == "r2"].groupby(
            ["stride", "space", "dim_group"]):
        v = sub.set_index("feature").value
        if not {"single", "pair"} <= set(v.index):
            continue
        shufv = float(v["pair_shuf"]) if "pair_shuf" in v.index else np.nan
        out.append(dict(stride=stride, space=space, dim_group=g,
                        single=float(v["single"]),
                        diff=float(v.get("diff", np.nan)),
                        pair_shuf=shufv,
                        pair=float(v["pair"]),
                        capacity_gain=shufv - float(v["single"]),
                        transition=float(v["pair"]) - shufv,
                        frac_state=float(v["single"] / v["pair"])
                        if v["pair"] > 0 else np.nan))
    return pd.DataFrame(out).sort_values(["dim_group", "stride", "space"])


def lam_reference():
    """LAM rows from exp 3, keyed by (stride, dim_group, feature)."""
    p = os.path.join(data.RESULTS_DIR, "exp3_lam.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    return d[d.metric == "r2"].set_index(["stride", "dim_group", "feature"]).value


def plot(df, path, lam=None):
    viz.setup()
    import matplotlib.pyplot as plt

    groups = ["xyz", "cont6"]
    glabel = {"xyz": "translation (xyz)", "cont6": "all 6 continuous"}
    r2 = df[df.metric == "r2"]
    spaces = [s for s in PLOT_ORDER if s in set(r2.space)]  # tolerate a partial run
    fig, axes = plt.subplots(len(groups), len(data.STRIDES),
                             figsize=(12.4, 7.4), sharey="row")
    kinds = [k for k in KINDS if k in set(r2.feature)]
    w = 0.86 / len(kinds) - 0.02
    off = (len(kinds) - 1) / 2.0

    for gi, g in enumerate(groups):
        for si, stride in enumerate(data.STRIDES):
            ax = axes[gi][si]
            x = np.arange(len(spaces))
            for ki, kind in enumerate(kinds):
                v = []
                for sp in spaces:
                    c = r2[(r2.stride == stride) & (r2.space == sp) &
                           (r2.feature == kind) & (r2.dim_group == g)].value
                    v.append(float(c.iloc[0]) if len(c) else np.nan)
                b = ax.bar(x + (ki - off) * (w + 0.02), v, w,
                           color=viz.SERIES[ki], linewidth=0, zorder=3,
                           label=KIND_LABEL[kind])
                viz.bar_labels(ax, b, dy=0.008)
            if lam is not None:
                # LAM rows are lines, not bars, and use the neutral inks so they
                # never collide with a series colour.  Their labels go in the
                # top-left corner rather than on the lines: the lines cross the
                # tallest bars, so an inline label would sit on top of them.
                for li, (feat, style, col, nm) in enumerate(
                        [("univla_lam_z_prevq", (0, (5, 2)), viz.INK, "pre-VQ"),
                         ("univla_lam_z_q", (0, (1.5, 2)), viz.MUTED,
                          "quantized")]):
                    try:
                        y = float(lam.loc[(stride, g, feat)])
                    except KeyError:
                        continue
                    ax.axhline(y, color=col, lw=1.3, ls=style, zorder=4)
                    ax.plot([0.02, 0.075], [0.975 - 0.062 * li] * 2, ls=style,
                            lw=1.3, color=col, transform=ax.transAxes,
                            clip_on=False, zorder=5)
                    ax.text(0.088, 0.975 - 0.062 * li,
                            f"UniVLA LAM {nm}  {y:.2f}", transform=ax.transAxes,
                            ha="left", va="center", fontsize=7.5, color=col,
                            zorder=5)
            ax.axhline(0, color=viz.BASELINE, lw=0.8, zorder=2)
            ax.set_xticks(x)
            ax.set_xticklabels([SPACE_LABEL[s] for s in spaces])
            # headroom for the two corner annotation lines
            ax.set_ylim(min(-0.05, r2[r2.dim_group == g].value.min() - 0.06), 1.12)
            ax.set_title(f"{glabel[g]}  —  stride {stride}")
            viz.grid(ax)
        axes[gi][0].set_ylabel("val $R^2$  (episode-level split)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(kinds),
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Exp 4 — CLS vs patch tokens, and how much of a two-frame "
                 "encoder's score is just single-frame state",
                 fontsize=11.5, fontweight="bold", color=viz.INK, y=1.0)
    fig.tight_layout()
    viz.save(fig, path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="clean")
    ap.add_argument("--spaces", nargs="+", default=list(SPACES))
    ap.add_argument("--kinds", nargs="+", default=list(KINDS))
    ap.add_argument("--strides", nargs="+", type=int, default=list(data.STRIDES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--budget-min", type=float, default=None,
                    help="stop before starting a cell that will not fit in the "
                         "remaining wall time; resubmit to continue")
    a = ap.parse_args()

    os.makedirs(data.RESULTS_DIR, exist_ok=True)
    os.makedirs(data.FIGS_DIR, exist_ok=True)
    out = os.path.join(data.RESULTS_DIR,
                       f"exp4_patch_matrix{probes.OUT_SUFFIX}.csv")

    pending = []
    if a.plot_only:
        df = pd.read_csv(out)
    else:
        print(f"[exp4] ridge alphas {ALPHAS}\n[exp4] logit Cs {probes.LOGIT_CS}")
        df, pending = run(a.condition, a.spaces, a.kinds, tuple(a.strides),
                          a.device, out_csv=out, budget_min=a.budget_min)
        df.to_csv(out, index=False)
        print(f"\n[exp4] wrote {out}")

    dec = decomposition(df)
    dp = os.path.join(data.RESULTS_DIR,
                      f"exp4_state_decomposition{probes.OUT_SUFFIX}.csv")
    dec.to_csv(dp, index=False)
    print(f"[exp4] wrote {dp}\n")
    print(dec[dec.dim_group.isin(["xyz", "cont6"])].to_string(index=False))

    plot(df, os.path.join(data.FIGS_DIR,
                          f"exp4_patch_matrix{probes.OUT_SUFFIX}.png"),
         lam_reference())
