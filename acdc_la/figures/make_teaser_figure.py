"""Teaser: figures/teaser.pdf -- the two challenges, one per panel, full width, page 1.

    ../envs/univla_vla/bin/python -m acdc_la.figures.make_teaser_figure

The large marks are schematic (they illustrate the two failure modes); the inset charts are numbers:
  (a) Profile NMSE = 1 - R^2(profile).  [real] endpoint-only .983 (s1_mean_1ep, prof .017),
      ours .909 (latp2_inv, prof .091).  LEDGER 2026-09-11 / 09-16.
  (b) effect error vs donor distance, Q1 and Q4.  [USER-SUPPLIED] .80 -> 1.06 (reconstruction only),
      .77 -> .86 (EAT), quoted from the manuscript's Fig. 3(a); not recomputed on this machine.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

INK, INK2, GRID = "#1a1a19", "#52514e", "#d9d9d6"
OURS, OURS_L, WARN = "#4a3aa7", "#a294db", "#eb6834"

PROFILE_NMSE = {"endpoint only": 0.983, "+ $A_1$": 0.909}          # [real]
QUART = {"reconstruction only": (0.80, 1.06), "EAT": (0.77, 0.86)}  # [USER-SUPPLIED]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.6, "figure.dpi": 160, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "text.color": INK,
})


def arrow(ax, p, q, color=INK2, lw=1.3, ls="-", ms=9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=4))


def tag(ax, t, title):
    ax.text(0, 104, f"({t})  {title}", fontsize=8.4, fontweight="bold", va="bottom", ha="left", color=INK)


def track(ax, y, xs, label, color=INK):
    """One clip: five frames on a common path, spaced by where the motion actually happened."""
    ax.plot([2, 38], [y, y], color=GRID, lw=1.2, zorder=1)
    for i, f in enumerate(xs):
        x = 2 + 36 * f
        ax.plot([x], [y], "o", ms=5.2, color=color if i in (0, len(xs) - 1) else "white",
                markeredgecolor=color, markeredgewidth=1.1, zorder=3)
    ax.text(2, y + 8, label, fontsize=7.4, color=INK, ha="left")


def panel_a(ax):
    tag(ax, "a", "Same endpoint, different process")

    track(ax, 74, [0.0, 0.45, 0.72, 0.90, 1.0], "fast, then slow", INK)
    track(ax, 42, [0.0, 0.10, 0.28, 0.55, 1.0], "slow, then fast", INK)

    ax.text(58, 92, "$N=\\sum_k \\Delta F_k$", fontsize=8.0, color=INK, ha="center")
    for y in (74, 42):
        arrow(ax, (47, y), (69, y), color=INK2, lw=1.6)
    ax.text(58, 57, "identical", fontsize=7.4, color=INK2, ha="center", style="italic")

    ax.text(86, 92, "$A_1$", fontsize=8.0, color=INK, ha="center")
    ax.plot([78, 94], [74, 74], color=INK2, lw=0.8, zorder=2)
    ax.plot([78, 94], [42, 42], color=INK2, lw=0.8, zorder=2)
    ax.add_patch(Rectangle((82, 74), 8, 11, facecolor=INK, edgecolor="white", lw=0.6, zorder=3))
    ax.add_patch(Rectangle((82, 31), 8, 11, facecolor=INK, edgecolor="white", lw=0.6, zorder=3))
    ax.text(86, 57, "opposite sign", fontsize=7.4, color=INK2, ha="center", style="italic")

    ax.text(58, 26, "one latent", fontsize=7.2, color=INK, ha="center")
    ax.text(86, 26, "separated", fontsize=7.2, color=INK, ha="center")


def panel_b(ax):
    tag(ax, "b", "One latent, a new initial state")
    box = dict(boxstyle="round,pad=0,rounding_size=1.6", linewidth=0.7)
    ax.add_patch(FancyBboxPatch((0, 50), 26, 17, facecolor=OURS_L, edgecolor=OURS, **box))
    ax.text(13, 58.5, "donor clip\n$\\rightarrow\\ z$", fontsize=7.4, ha="center", va="center", color=INK,
            linespacing=1.3)
    ax.text(13, 44, "same $z$,\ntwo recipients", fontsize=6.9, ha="center", va="top", color=INK2, linespacing=1.3)

    ax.text(52, 96, "near state", fontsize=7.2, color=INK2, ha="center")
    ax.text(84, 96, "far state", fontsize=7.2, color=INK2, ha="center")
    import math
    for r, (name, col, angles) in enumerate([("reconstruction only", WARN, (16, 54)),
                                             ("EAT", OURS, (5, 8))]):
        y = 76 - r * 44
        ax.text(38, y + 13, name, fontsize=7.4, color=col, ha="left", fontweight="bold")
        arrow(ax, (26, 58.5), (36, y), color=OURS, lw=1.0)
        for c in (0, 1):
            x = 38 + c * 32
            ax.add_patch(FancyBboxPatch((x, y - 10), 28, 20, facecolor="white", edgecolor=GRID, **box))
            arrow(ax, (x + 5, y - 4), (x + 24, y - 4), color=GRID, lw=1.0, ls=(0, (2, 2)), ms=7)
            a = math.radians(angles[c])
            arrow(ax, (x + 5, y - 4), (x + 5 + 19 * math.cos(a), y - 4 + 19 * math.sin(a)), color=col, lw=1.5)




def inset_a(ax):
    ins = ax.inset_axes([0.12, -0.26, 0.50, 0.24])
    ks = list(PROFILE_NMSE); v = [PROFILE_NMSE[k] for k in ks]
    b = ins.barh([1, 0], v, height=0.55, color=[OURS_L, OURS], edgecolor="white", lw=0.7, zorder=3)
    for bb, vv in zip(b, v):
        ins.text(vv - 0.003, bb.get_y() + bb.get_height() / 2, f"{vv:.3f}", va="center", ha="right",
                 fontsize=6.6, color="white")
    ins.set_yticks([1, 0]); ins.set_yticklabels(ks, fontsize=6.8, color=INK2)
    ins.set_xlim(0.86, 1.0); ins.set_xticks([])
    ins.set_xlabel("profile NMSE   (lower is better)", fontsize=6.6, color=INK2, labelpad=2)
    for sp in ("top", "right", "bottom"): ins.spines[sp].set_visible(False)
    ins.spines["left"].set_color(INK2); ins.spines["left"].set_linewidth(0.6)
    ins.tick_params(length=0, pad=1.5)


def inset_b(ax):
    ins = ax.inset_axes([0.16, -0.26, 0.42, 0.24])
    for (k, (q1, q4)), c in zip(QUART.items(), (WARN, OURS)):
        ins.plot([0, 1], [q1, q4], "-o", ms=4.2, lw=1.4, color=c, markeredgecolor="white", markeredgewidth=0.7)
        ins.text(1.08, q4, "recon. only" if "recon" in k else k, fontsize=6.6, color=c, va="center")
    ins.set_xticks([0, 1]); ins.set_xticklabels(["Q1", "Q4"], fontsize=6.8, color=INK2)
    ins.set_xlim(-0.12, 1.1); ins.set_ylim(0.70, 1.14)
    ins.set_yticks([0.8, 1.0]); ins.tick_params(labelsize=6.6, length=2, pad=1.5, colors=INK2)
    ins.set_xlabel("donor distance quartile", fontsize=6.6, color=INK2, labelpad=1.5)
    ins.set_ylabel("effect error", fontsize=6.6, color=INK2, labelpad=1.5)
    ins.yaxis.grid(True, color=GRID, lw=0.5); ins.set_axisbelow(True)
    for sp in ("top", "right"): ins.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ins.spines[sp].set_color(INK2); ins.spines[sp].set_linewidth(0.6)


def teaser(out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"wspace": 0.16})
    for ax in axes:
        ax.set_xlim(-1, 101); ax.set_ylim(18, 100); ax.axis("off")
    panel_a(axes[0]); inset_a(axes[0])
    panel_b(axes[1]); inset_b(axes[1])
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png"); plt.close(fig)


if __name__ == "__main__":
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "figures")
    teaser(os.path.join(d, "teaser"))
    print("wrote", d + "/teaser.{pdf,png}")
