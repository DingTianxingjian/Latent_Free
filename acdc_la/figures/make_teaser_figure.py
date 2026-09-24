"""Teaser: figures/teaser.{pdf,png} -- the two challenges, full width, page 1.

    ../envs/univla_vla/bin/python -m acdc_la.figures.make_teaser_figure

Panel (a) is MEASURED, not drawn.  One real pair of OXE validation clips from the same source with
  cos(N) = 0.86   (the same net effect)      cos(A1) = -0.46   (opposite first dynamics mode)
Its two curves are the clips' own action labels (per-segment xyz displacement, interpolated uniformly in
time, so dot spacing IS speed), and the two latent panels are a 2-D PCA of that source's 400 latents under
each tokenizer, with the pair marked.  Numbers: LEDGER 2026-09-24, dumps s1_mean_1ep_val / latp2_inv_val.
Panel (b) is a schematic of the transport setting, in the layout of the manuscript's own teaser.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, INK2, GRID = "#1a1a19", "#52514e", "#d9d9d6"
FAST, SLOW = "#2a78d6", "#eb6834"          # the two clips: "fast then slow" / "slow then fast"
OURS, OURS_L, WARN = "#4a3aa7", "#e7e3f5", "#eb6834"
PAIR = dict(i=7247, j=7444, src=18, cosN=0.86, cosA1=-0.46, cos_endpoint=0.895, cos_ours=0.181,
            n_pairs=1111, agg_endpoint=0.628, agg_ours=0.416)
ACT = {                                     # [real] the two clips' action labels, (4 segments, xyz)
    "j": [[0.0373, 0.0555, -0.1715], [0.0046, 0.1095, -0.0145], [0.0005, -0.0038, -0.0071], [0.0028, -0.0021, -0.0189]],
    "i": [[0.0250, 0.0291, 0.0036], [0.0094, 0.0323, -0.1098], [0.0407, 0.0789, -0.0744], [-0.0109, 0.0520, -0.0040]],
}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.6, "figure.dpi": 160, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "text.color": INK,
})


def _path(act, n_sub=7):
    """Cumulative end-effector path, sampled uniformly in time: dot spacing is speed."""
    a = np.asarray(act, np.float32)
    knots = np.vstack([np.zeros(3, np.float32), np.cumsum(a, 0)])
    pts = [knots[k] + (knots[k + 1] - knots[k]) * t for k in range(len(a)) for t in np.linspace(0, 1, n_sub, endpoint=False)]
    return np.array(pts + [knots[-1]])


def panel_a_frames(fig, gs, d):
    """Two real clips of the same source: three frames each, with their own speed profile."""
    import numpy as _np
    for r, (k, col, lab) in enumerate((("j", FAST, "fast $\\rightarrow$ slow"), ("i", SLOW, "slow $\\rightarrow$ fast"))):
        clip = _np.load(os.path.join(d, f"teaser_clip_{k}.npy"))                 # (5, 256, 256, 3) uint8
        spd = _np.linalg.norm(_np.asarray(ACT[k], _np.float32), axis=1)          # real per-segment displacement
        for c, fi in enumerate((0, 2, 4)):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(clip[fi]); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_color(col if c else GRID); sp.set_linewidth(0.9 if c else 0.6)
            if r == 0: ax.set_title(f"$t{{=}}{fi}$", fontsize=6.2, color=INK2, pad=2)
            if c == 0:
                ax.text(-0.06, 0.5, lab, transform=ax.transAxes, rotation=90, va="center", ha="center",
                        fontsize=7.0, color=col, fontweight="bold")
        ax = fig.add_subplot(gs[r, 3])
        ax.plot(range(4), spd, "-o", color=col, lw=1.2, ms=3.4, markeredgecolor="white", markeredgewidth=0.6)
        ax.set_ylim(0, max(_np.linalg.norm(_np.asarray(ACT["i"], _np.float32), axis=1).max(),
                           _np.linalg.norm(_np.asarray(ACT["j"], _np.float32), axis=1).max()) * 1.2)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(GRID); ax.spines[sp].set_linewidth(0.6)
        if r == 0: ax.set_title("speed", fontsize=6.2, color=INK2, pad=2)


def panel_a(ax_tr, ax_e, ax_o, cache):
    if ax_tr is None:
        _latents(ax_e, ax_o, cache); return
    P = {k: _path(v) for k, v in ACT.items()}
    B = np.vstack(list(P.values()))
    _, _, Vt = np.linalg.svd(B - B.mean(0), full_matrices=False)          # one 2-D view for both clips
    for k, col, lab, y in (("j", FAST, "fast $\\rightarrow$ slow", 1), ("i", SLOW, "slow $\\rightarrow$ fast", 0)):
        q = (P[k] - B.mean(0)) @ Vt[:2].T
        q = q - q[0] + np.array([0.0, y * 0.30])
        ax_tr.plot(q[:, 0], q[:, 1], "-", color=col, lw=0.9, alpha=0.5, zorder=2)
        ax_tr.scatter(q[:, 0], q[:, 1], s=7.5, color=col, zorder=3, linewidths=0)
        ax_tr.add_patch(FancyArrowPatch(q[0], q[-1], arrowstyle="-|>", mutation_scale=8, color=col, lw=1.0,
                                        linestyle=(0, (3, 2)), shrinkA=0, shrinkB=0, zorder=4, alpha=0.85))
        ax_tr.text(0.0, 0.97 if y else 0.40, lab, transform=ax_tr.transAxes, color=col, fontsize=7.4,
                   ha="left", va="top")
    ax_tr.text(0.5, -0.10, f"same net effect  $\\cos(N)={PAIR['cosN']:.2f}$,  opposite dynamics  "
                           f"$\\cos(A_1)={PAIR['cosA1']:+.2f}$", transform=ax_tr.transAxes, ha="center",
               va="top", fontsize=6.8, color=INK2)
    ax_tr.set_aspect("equal"); ax_tr.axis("off")

    _latents(ax_e, ax_o, cache)


def _latents(ax_e, ax_o, cache):
    for ax, key, title, cos in ((ax_e, "endpoint", "endpoint only", PAIR["cos_endpoint"]),
                                (ax_o, "ours", "net $+\\ A_1$", PAIR["cos_ours"])):
        c = cache[key]; A = np.array(c["all"])
        ax.scatter(A[:, 0], A[:, 1], s=3.5, color=GRID, zorder=1, linewidths=0)
        for cl, col in (("j", FAST), ("i", SLOW)):
            p = np.array(c[cl])
            ax.plot([p[0]], [p[1]], "o", ms=7.5, color=col, markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        pi, pj = np.array(c["i"]), np.array(c["j"])
        ax.plot([pi[0], pj[0]], [pi[1], pj[1]], color=INK2, lw=0.8, ls=(0, (2, 2)), zorder=2)
        ax.set_title(title, fontsize=7.0, color=INK, pad=3)
        ax.text(0.5, -0.06, f"$\\cos(z)={cos:.2f}$", transform=ax.transAxes, ha="center", va="top",
                fontsize=7.4, color=OURS if key == "ours" else INK2,
                fontweight="bold" if key == "ours" else "normal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color(GRID); s.set_linewidth(0.6)
        ax.set_aspect("equal")


def panel_b(ax):
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    box = dict(boxstyle="round,pad=0,rounding_size=1.6", linewidth=0.7)
    ax.add_patch(FancyBboxPatch((0, 46), 24, 16, facecolor="white", edgecolor=GRID, **box))
    ax.text(12, 54, "donor transition", fontsize=7.2, ha="center", va="center", color=INK)
    ax.add_patch(FancyArrowPatch((6, 44), (20, 44), arrowstyle="-|>", mutation_scale=7, color=INK2,
                                 lw=1.0, linestyle=(0, (3, 2)), shrinkA=0, shrinkB=0))
    ax.text(13, 38, "source effect", fontsize=6.6, color=INK2, ha="center")
    ax.add_patch(plt.Circle((32, 54), 5.5, facecolor=OURS_L, edgecolor=OURS, lw=0.8, zorder=3))
    ax.text(32, 54, "$z$", fontsize=8, ha="center", va="center", color=OURS, zorder=4)
    ax.add_patch(FancyArrowPatch((24, 54), (26.5, 54), arrowstyle="-|>", mutation_scale=7, color=INK2, lw=0.9))

    for r, (name, col, ang) in enumerate([("reconstruction only", WARN, (18, 58)), ("EAT (ours)", OURS, (6, 9))]):
        y = 78 - r * 46
        ax.add_patch(FancyArrowPatch((37, 54), (48, y - 4 + 8), arrowstyle="-|>", mutation_scale=7,
                                     color=INK2, lw=0.9, connectionstyle=f"arc3,rad={0.18 if r else -0.18}"))
        ax.text(52, y + (13 if r == 0 else -15), name, fontsize=7.2, color=col, ha="left", fontweight="bold")
        for c, lab in enumerate(("similar state", "shifted state")):
            x = 52 + c * 25
            ax.add_patch(FancyBboxPatch((x, y - 12), 22, 20, facecolor="white", edgecolor=GRID, **box))
            if r == 0:
                ax.text(x + 11, y + 24, lab, fontsize=6.8, color=INK2, ha="center")
            ax.add_patch(FancyArrowPatch((x + 4, y - 4), (x + 19, y - 4), arrowstyle="-|>", mutation_scale=6,
                                         color=GRID, lw=1.0, linestyle=(0, (2, 2)), shrinkA=0, shrinkB=0))
            a = np.radians(ang[c])
            ax.add_patch(FancyArrowPatch((x + 4, y - 4), (x + 4 + 15 * np.cos(a), y - 4 + 15 * np.sin(a)),
                                         arrowstyle="-|>", mutation_scale=7, color=col, lw=1.4, shrinkA=0, shrinkB=0))
    ax.add_patch(FancyArrowPatch((36, 4), (44, 4), arrowstyle="-|>", mutation_scale=6, color=GRID, lw=1.0,
                                 linestyle=(0, (2, 2)), shrinkA=0, shrinkB=0))
    ax.text(45.5, 4, "source effect", fontsize=6.4, color=INK2, va="center")
    ax.add_patch(FancyArrowPatch((72, 4), (80, 4), arrowstyle="-|>", mutation_scale=6, color=OURS, lw=1.2,
                                 shrinkA=0, shrinkB=0))
    ax.text(81.5, 4, "decoded effect", fontsize=6.4, color=INK2, va="center")


def teaser(out, cache, d):
    fig = plt.figure(figsize=(7.2, 2.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.7, 1.0, 2.2], wspace=0.22)
    gsa = gs[0].subgridspec(2, 4, width_ratios=[1, 1, 1, 0.75], wspace=0.07, hspace=0.16)
    gsl = gs[1].subgridspec(1, 2, wspace=0.34)
    ax_e, ax_o = fig.add_subplot(gsl[0]), fig.add_subplot(gsl[1])
    ax_b = fig.add_subplot(gs[2])
    panel_a_frames(fig, gsa, d)
    panel_a(None, ax_e, ax_o, cache)
    panel_b(ax_b)
    fig.text(0.02, 1.10, "(a)  What to keep?", fontsize=9, fontweight="bold", va="bottom", color=INK)
    fig.text(0.02, 1.02, "Same endpoint, different process.", fontsize=7.4, va="bottom", color=INK2)
    fig.text(0.575, 1.10, "(b)  How to behave?", fontsize=9, fontweight="bold", va="bottom", color=INK)
    fig.text(0.575, 1.02, "Same latent, different initial states.", fontsize=7.4, va="bottom", color=INK2)
    fig.text(0.26, -0.085, f"one real validation pair; over all {PAIR['n_pairs']:,} such pairs "
             f"$\\cos(z)$ falls {PAIR['agg_endpoint']:.2f} $\\rightarrow$ {PAIR['agg_ours']:.2f}",
             ha="center", fontsize=6.6, color=INK2, style="italic")
    fig.savefig(out + ".pdf", pad_inches=0.06); fig.savefig(out + ".png", pad_inches=0.06); plt.close(fig)


if __name__ == "__main__":
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "figures")
    cache = json.load(open(os.path.join(d, "teaser_latents.json")))
    teaser(os.path.join(d, "teaser"), cache, d)
    print("wrote", d + "/teaser.{pdf,png}")
