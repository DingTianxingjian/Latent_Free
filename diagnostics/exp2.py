"""Experiment 2 -- distractor contamination, and the link back to the theory.

Part A  re-runs the Exp-1 probe inside each corrupted condition (train and eval
        within the condition), so we see how much action information survives.

Part B  is the actual theory test.  "LAM latents ~ PCA of frame differences"
        predicts that a LAM bottleneck keeps the *largest-variance* directions of
        Delta-f.  So we ask: does the action-carrying subspace still live inside
        the top-k principal components once a distractor is present?

          * action subspace A : the span of the 6 ridge weight vectors fitted on
            CLEAN data, expressed in raw (unstandardized) feature coordinates and
            orthonormalized.  This is the set of feature directions a linear
            action decoder actually reads.
          * PC subspace P_k   : top-k PCs of Delta-f for a given condition, fitted
            on the train split of that condition.
          * retained energy   : ||P_k A||_F^2 / 6, i.e. the mean squared cosine of
            the principal angles between the two subspaces.  1.0 means a k-dim
            bottleneck keeps everything the decoder needs; 0.0 means it throws
            all of it away.

        Both subspaces live in the same 768-d DINOv2 space, so the comparison is
        well posed.  Raw (not standardized) coordinates are used on purpose: the
        theory is about variance in the encoder's own feature space, and
        standardization is only a probe-fitting convenience.
"""

import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import data
import probes
import viz
from features import CONDITIONS

KS = (1, 2, 4, 8, 16, 32, 64, 128)
REPORT_KS = (8, 16, 32)
COND_LABEL = {
    "clean": "clean",
    "patch": "patch\n(moving)",
    "patch_static": "patch\n(frozen)",
    "border": "border\n(playing)",
    "border_static": "border\n(frozen)",
}


def load_feats(cond):
    return np.load(os.path.join(data.CACHE_DIR, f"feats_{cond}.npy"))


# --------------------------------------------------------------------------- #
# Part A -- probes within each condition
# --------------------------------------------------------------------------- #
def probe_all(conditions, strides=data.STRIDES):
    index = data.build_index()
    rows, weights = [], {}
    for cond in conditions:
        feats = load_feats(cond)
        for stride in strides:
            itr, jtr, yc_tr, yg_tr = data.build_pairs(index, stride, "train")
            iva, jva, yc_va, yg_va = data.build_pairs(index, stride, "val")
            for kind in ("diff", "single"):
                Ftr = probes.make_features(feats, itr, jtr, kind)
                Fva = probes.make_features(feats, iva, jva, kind)
                res, W_raw = probes.run_probe_set(Ftr, Fva, yc_tr, yc_va,
                                                  yg_tr, yg_va)
                if kind == "diff":
                    weights[(cond, stride)] = W_raw
                print(f"[exp2:A] {cond:14s} s={stride} {kind:6s} "
                      f"R2 xyz={res['r2_xyz']:.3f} rot={res['r2_rot']:.3f} "
                      f"cont6={res['r2_cont6']:.3f} "
                      f"grip={res['bal_acc_grip']:.3f}", flush=True)
                for g in ("xyz", "rot", "cont6"):
                    rows.append(dict(condition=cond, stride=stride, feature=kind,
                                     dim_group=g, metric="r2",
                                     value=res[f"r2_{g}"],
                                     hyperparam=f"alpha={res['alpha']:g}",
                                     n_train=res["n_train"], n_val=res["n_val"]))
                rows.append(dict(condition=cond, stride=stride, feature=kind,
                                 dim_group="gripper", metric="balanced_accuracy",
                                 value=res["bal_acc_grip"],
                                 hyperparam=f"C={res['C']:g}",
                                 n_train=res["n_train"], n_val=res["n_val"]))
        del feats
    return pd.DataFrame(rows), weights


# --------------------------------------------------------------------------- #
# Part B -- PCA of Delta-f vs the clean action subspace
# --------------------------------------------------------------------------- #
def action_basis(W_raw):
    """Orthonormal basis (D,6) for the span of the 6 ridge weight vectors."""
    Q, _ = np.linalg.qr(W_raw.T)
    assert Q.shape[1] == 6, Q.shape
    return Q


def pca_all(conditions, weights, strides=data.STRIDES, ks=KS):
    index = data.build_index()
    rows = []
    for stride in strides:
        A = action_basis(weights[("clean", stride)])       # clean action subspace
        itr, jtr, _, _ = data.build_pairs(index, stride, "train")
        for cond in conditions:
            feats = load_feats(cond)
            D = feats[jtr] - feats[itr]                     # Delta-f, train only
            p = PCA(n_components=max(ks), svd_solver="full", random_state=0)
            p.fit(D)
            evr = np.cumsum(p.explained_variance_ratio_)
            for k in ks:
                Pk = p.components_[:k]                      # (k,D), orthonormal
                energy = float(np.sum((Pk @ A) ** 2) / A.shape[1])
                rows.append(dict(stride=stride, condition=cond, k=k,
                                 cum_var_explained=float(evr[k - 1]),
                                 action_energy_in_topk=energy))
                if k in REPORT_KS:
                    print(f"[exp2:B] s={stride} {cond:14s} k={k:3d} "
                          f"var={evr[k-1]:.3f} action_energy={energy:.3f}",
                          flush=True)
            del feats, D
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def plot_r2(df, conditions, path):
    viz.setup()
    import matplotlib.pyplot as plt

    groups = ["xyz", "rot", "cont6"]
    glab = ["translation xyz", "rotation", "all 6 continuous"]
    d = df[(df.feature == "diff")]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 6.4),
                             gridspec_kw=dict(height_ratios=[1.35, 1]))
    x = np.arange(len(conditions))
    w = 0.26
    lo = min(0.0, d[d.dim_group.isin(groups)].value.min())

    for c, stride in enumerate(data.STRIDES):
        ax = axes[0, c]
        for s, g in enumerate(groups):
            v = [float(d[(d.stride == stride) & (d.condition == cond) &
                         (d.dim_group == g)].value.iloc[0]) for cond in conditions]
            b = ax.bar(x + (s - 1) * (w + 0.02), v, w, color=viz.SERIES[s],
                       linewidth=0, zorder=3, label=glab[s])
            viz.bar_labels(ax, b)
        ax.axhline(0, color=viz.BASELINE, lw=0.8, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABEL[c_] for c_ in conditions])
        ax.set_ylim(lo - 0.08, 1.0)
        ax.set_title(f"stride {stride} — continuous dims")
        viz.grid(ax)
        if c == 0:
            ax.set_ylabel(r"val $R^2$ from $\Delta f$")

        ax = axes[1, c]
        v = [float(d[(d.stride == stride) & (d.condition == cond) &
                     (d.dim_group == "gripper")].value.iloc[0])
             for cond in conditions]
        b = ax.bar(x, v, 0.5, color=viz.INK2, linewidth=0, zorder=3)
        viz.bar_labels(ax, b, dy=0.006)
        ax.axhline(0.5, color=viz.MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.annotate("chance", (ax.get_xlim()[1], 0.5), xytext=(-2, -4),
                    textcoords="offset points", ha="right", va="top",
                    fontsize=7.5, color=viz.MUTED)
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABEL[c_] for c_ in conditions])
        ax.set_ylim(0.4, 1.02)
        ax.set_title(f"stride {stride} — gripper")
        viz.grid(ax)
        if c == 0:
            ax.set_ylabel("val balanced accuracy")

    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Exp 2 — action decodability under image-level distractors "
                 "(probe trained and evaluated within each condition)",
                 fontsize=11, fontweight="bold", color=viz.INK, y=1.0)
    fig.tight_layout()
    viz.save(fig, path)


def plot_pca(df, conditions, path):
    viz.setup()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 6.8))
    for r, stride in enumerate(data.STRIDES):
        for c, (col, ylab, ttl) in enumerate([
            ("action_energy_in_topk",
             "fraction of clean action\nsubspace inside top-k PCs",
             "action-subspace energy retained"),
            ("cum_var_explained", "cumulative variance explained",
             r"variance of $\Delta f$ captured")]):
            ax = axes[r, c]
            if col == "action_energy_in_topk":
                # what a RANDOM k-dim subspace would retain (k/D).  Without this
                # reference the panel just looks "small"; the actual finding is
                # that the curves sit an order of magnitude *below* chance.
                ks = sorted(df.k.unique())
                ax.plot(ks, [k / 768 for k in ks], color=viz.MUTED, lw=1.4,
                        ls=(0, (4, 3)), zorder=2, label="chance (random k-dim)")
            for s, cond in enumerate(conditions):
                sub = df[(df.stride == stride) &
                         (df.condition == cond)].sort_values("k")
                ax.plot(sub.k, sub[col], color=viz.SERIES[s], lw=2.0,
                        marker=viz.MARKERS[s], ms=5, zorder=3,
                        label=COND_LABEL[cond].replace("\n", " "))
            ax.set_xscale("log", base=2)
            ax.set_xticks(list(KS))
            ax.set_xticklabels([str(k) for k in KS])
            ax.set_xlabel("k  (latent-bottleneck width)")
            if col == "action_energy_in_topk":
                # zoom to the data + chance band; 0-1 would squash it flat
                ax.set_ylim(0, max(df.k) / 768 * 1.2)
            else:
                ax.set_ylim(0, 1.02)
            ax.set_title(f"stride {stride} — {ttl}")
            viz.grid(ax, axis="both")
            if c == 0:
                ax.set_ylabel(ylab)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Exp 2 — does a top-k (PCA-like) bottleneck still contain the "
                 "action directions?", fontsize=11, fontweight="bold",
                 color=viz.INK, y=1.0)
    fig.tight_layout()
    viz.save(fig, path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true",
                    help="regenerate figures from existing CSVs (no refitting)")
    args = ap.parse_args()

    os.makedirs(data.RESULTS_DIR, exist_ok=True)
    os.makedirs(data.FIGS_DIR, exist_ok=True)
    conds = [c for c in CONDITIONS
             if os.path.exists(os.path.join(data.CACHE_DIR, f"feats_{c}.npy"))]
    print(f"[exp2] conditions available: {conds}")

    if args.plot_only:
        df_a = pd.read_csv(os.path.join(
            data.RESULTS_DIR, f"exp2_distractor{probes.OUT_SUFFIX}.csv"))
        df_b = pd.read_csv(os.path.join(
            data.RESULTS_DIR, f"exp2_pca_action_energy{probes.OUT_SUFFIX}.csv"))
        conds = [c for c in CONDITIONS if c in set(df_a.condition)]
        plot_r2(df_a, conds, os.path.join(
            data.FIGS_DIR, f"exp2_r2_clean_vs_distracted{probes.OUT_SUFFIX}.png"))
        plot_pca(df_b, conds, os.path.join(
            data.FIGS_DIR, f"exp2_pca_action_energy{probes.OUT_SUFFIX}.png"))
        raise SystemExit(0)

    df_a, weights = probe_all(conds)
    out_a = os.path.join(data.RESULTS_DIR,
                         f"exp2_distractor{probes.OUT_SUFFIX}.csv")
    df_a.to_csv(out_a, index=False)
    print(f"[exp2] wrote {out_a}")

    df_b = pca_all(conds, weights)
    out_b = os.path.join(data.RESULTS_DIR,
                         f"exp2_pca_action_energy{probes.OUT_SUFFIX}.csv")
    df_b.to_csv(out_b, index=False)
    print(f"[exp2] wrote {out_b}")

    plot_r2(df_a, conds, os.path.join(
        data.FIGS_DIR, f"exp2_r2_clean_vs_distracted{probes.OUT_SUFFIX}.png"))
    plot_pca(df_b, conds, os.path.join(
        data.FIGS_DIR, f"exp2_pca_action_energy{probes.OUT_SUFFIX}.png"))
