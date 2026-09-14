"""Experiment 1 -- how much ground-truth action is linearly decodable from the
frame-difference of a frozen generic encoder?

This is the reference ceiling for what any LAM (which compresses exactly this
Delta-f signal) could possibly contain.

The `single` ablation probes f(o_t) alone.  Whatever it recovers is trivial
state correlation -- in scripted demos the action at time t is highly
predictable from the pose at time t -- not transition information.  The gap
between `diff` and `single` is the part that genuinely needs the transition.
"""

import os

import numpy as np
import pandas as pd

import data
import probes
import viz

FEATURE_KINDS = ("diff", "single")
KIND_LABEL = {"diff": r"$\Delta f$  (frame difference)",
              "single": r"$f(o_t)$  (single frame, ablation)"}


def run(condition="clean", strides=data.STRIDES, save_weights=True):
    """Probe every (stride, feature-kind) combination.  Returns a tidy frame."""
    index = data.build_index()
    feats = np.load(os.path.join(data.CACHE_DIR, f"feats_{condition}.npy"))
    print(f"[exp1] features {condition}: {feats.shape}")

    rows = []
    for stride in strides:
        itr, jtr, yc_tr, yg_tr = data.build_pairs(index, stride, "train")
        iva, jva, yc_va, yg_va = data.build_pairs(index, stride, "val")
        print(f"[exp1] stride {stride}: {len(itr)} train / {len(iva)} val pairs")

        for kind in FEATURE_KINDS:
            Ftr = probes.make_features(feats, itr, jtr, kind)
            Fva = probes.make_features(feats, iva, jva, kind)
            res, W_raw = probes.run_probe_set(Ftr, Fva, yc_tr, yc_va,
                                              yg_tr, yg_va)
            if save_weights and kind == "diff":
                np.save(os.path.join(data.CACHE_DIR,
                                     f"Wact_{condition}_s{stride}.npy"), W_raw)
            print(f"   {kind:6s} alpha={res['alpha']:<6g} C={res['C']:<6g} "
                  f"R2 xyz={res['r2_xyz']:.3f} rot={res['r2_rot']:.3f} "
                  f"cont6={res['r2_cont6']:.3f} grip_bacc={res['bal_acc_grip']:.3f}")

            for group in ("xyz", "rot", "cont6"):
                rows.append(dict(condition=condition, stride=stride,
                                 feature=kind, dim_group=group, metric="r2",
                                 value=res[f"r2_{group}"],
                                 hyperparam=f"alpha={res['alpha']:g}",
                                 n_train=res["n_train"], n_val=res["n_val"]))
            rows.append(dict(condition=condition, stride=stride, feature=kind,
                             dim_group="gripper", metric="balanced_accuracy",
                             value=res["bal_acc_grip"],
                             hyperparam=f"C={res['C']:g}",
                             n_train=res["n_train"], n_val=res["n_val"]))
            for k in range(6):
                rows.append(dict(condition=condition, stride=stride,
                                 feature=kind,
                                 dim_group=["dx", "dy", "dz", "drx", "dry", "drz"][k],
                                 metric="r2", value=res[f"r2_dim{k}"],
                                 hyperparam=f"alpha={res['alpha']:g}",
                                 n_train=res["n_train"], n_val=res["n_val"]))
    return pd.DataFrame(rows)


def plot(df, path):
    viz.setup()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9),
                             gridspec_kw=dict(width_ratios=[1, 1, 0.72]))
    groups = ["xyz", "rot", "cont6"]
    glabel = ["translation\nxyz", "rotation\n(axis-angle)", "all 6\ncontinuous"]
    w = 0.36

    r2 = df[df.metric == "r2"]
    lo = min(0.0, r2[r2.dim_group.isin(groups)].value.min())

    for ax, stride in zip(axes[:2], data.STRIDES):
        x = np.arange(len(groups))
        for s, kind in enumerate(FEATURE_KINDS):
            v = [float(r2[(r2.stride == stride) & (r2.feature == kind) &
                          (r2.dim_group == g)].value.iloc[0]) for g in groups]
            # 2px surface gap between adjacent bars
            b = ax.bar(x + (s - 0.5) * (w + 0.02), v, w, color=viz.SERIES[s],
                       linewidth=0, zorder=3, label=KIND_LABEL[kind])
            viz.bar_labels(ax, b)
        ax.axhline(0, color=viz.BASELINE, lw=0.8, zorder=2)
        ax.set_xticks(x); ax.set_xticklabels(glabel)
        ax.set_ylim(lo - 0.08, 1.0)
        ax.set_title(f"stride {stride}  ({'1 step' if stride == 1 else '4 steps'})")
        viz.grid(ax)
    axes[0].set_ylabel("val $R^2$  (episode-level split)")

    ax = axes[2]
    x = np.arange(len(data.STRIDES))
    ba = df[df.dim_group == "gripper"]
    for s, kind in enumerate(FEATURE_KINDS):
        v = [float(ba[(ba.stride == st) & (ba.feature == kind)].value.iloc[0])
             for st in data.STRIDES]
        b = ax.bar(x + (s - 0.5) * (w + 0.02), v, w, color=viz.SERIES[s],
                   linewidth=0, zorder=3)
        viz.bar_labels(ax, b, dy=0.006)
    ax.axhline(0.5, color=viz.MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.annotate("chance", (ax.get_xlim()[1], 0.5), xytext=(-2, -4),
                textcoords="offset points", ha="right", va="top",
                fontsize=7.5, color=viz.MUTED)
    ax.set_xticks(x); ax.set_xticklabels([f"stride {s}" for s in data.STRIDES])
    ax.set_ylim(0.4, 1.0)
    ax.set_title("gripper")
    ax.set_ylabel("val balanced accuracy")
    viz.grid(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Exp 1 — action decodability from a frozen DINOv2 ViT-B/14 "
                 "(LIBERO-spatial, clean frames)",
                 fontsize=11, fontweight="bold", color=viz.INK, y=1.03)
    viz.save(fig, path)


if __name__ == "__main__":
    os.makedirs(data.RESULTS_DIR, exist_ok=True)
    os.makedirs(data.FIGS_DIR, exist_ok=True)
    print(f"[exp1] ridge alphas {probes.RIDGE_ALPHAS}  logit Cs {probes.LOGIT_CS}")
    df = run("clean")
    out = os.path.join(data.RESULTS_DIR,
                       f"exp1_decodability{probes.OUT_SUFFIX}.csv")
    df.to_csv(out, index=False)
    print(f"[exp1] wrote {out}")

    main = df[df.dim_group.isin(["xyz", "rot", "cont6", "gripper"])]
    print("\n" + main.to_string(index=False))
    plot(df, os.path.join(data.FIGS_DIR,
                          f"exp1_r2_bars{probes.OUT_SUFFIX}.png"))

    top = df[(df.feature == "diff") & (df.dim_group == "cont6")].value.max()
    if top < 0.05:
        print("\n*** WARNING: continuous R^2 ~ 0 for Delta-f. Suspect a pipeline "
              "bug (frame/action alignment, normalization) before concluding "
              "anything. ***")
