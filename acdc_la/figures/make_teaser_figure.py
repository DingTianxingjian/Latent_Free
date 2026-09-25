"""Teaser: figures/teaser.{pdf,png} -- the two counter-examples, full width, page 1.

    ../envs/univla_vla/bin/python -m acdc_la.figures.make_teaser_figure

Visual language after Jenni et al. (ECCV 2020), SpeedNet (CVPR 2020) and MoCoGAN (CVPR 2018): data first,
few and large frames, the text in the caption.

(a) Same endpoints, different dynamics.  ONE real LIBERO-Spatial demonstration (approach phase, t = 0..61)
    resampled under two temporal warps, w(u) = 1-(1-u)^2 and w(u) = u^2, four frames each.  First and last
    frames are the same pixels by construction, so an endpoint encoder must return the same latent.
(b) Same action, different states.  Real LIBERO-Spatial scenes (donor, a similar and a shifted recipient),
    one t = 0 frame per cell.  The arrows are SCHEMATIC effect directions drawn on real frames.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

INK, INK2, GRID = "#1a1a19", "#52514e", "#d9d9d6"
FAST, SLOW = "#2a78d6", "#eb6834"
OURS, BAD, OK, RECON = "#4a3aa7", "#d92d20", "#1baf7a", "#eb6834"

EPS = "figures/teaser_eps/keep"
DONOR, SIMILAR, SHIFTED = "table_center__libero_spatial_0", "table_center__libero_spatial_1", \
                          "next_to_the_cookie_box__libero_spatial_0"
T_GRASP = 61                                              # the donor's grasp frame: end of the approach phase
GRIP = (0.50, 0.27)                                       # gripper at t = 0, frame fractions from top-left
BOWL = {DONOR: (0.48, 0.62), SIMILAR: (0.48, 0.62), SHIFTED: (0.62, 0.83)}
CROP_A = (0.20, 0.10, 0.80, 0.70)                         # (a): arm and bowl fill the frame
CROP_B = (0.12, 0.14, 0.88, 0.90)                         # (b): wide enough to hold the shifted bowl
N_STRIP = 4

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.4, "figure.dpi": 160, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "text.color": INK,
})


def _frames(tag):
    return np.load(os.path.join(EPS, tag + ".npy"))


def _img(ax, im, crop, edge=GRID, lw=0.6):
    H, W = im.shape[:2]
    ax.imshow(im[int(crop[1] * H):int(crop[3] * H), int(crop[0] * W):int(crop[2] * W)])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(edge); s.set_linewidth(lw)


def _pt(p, crop, n=256):
    side = (crop[2] - crop[0]) * n
    return ((p[0] - crop[0]) / (crop[2] - crop[0]) * side, (p[1] - crop[1]) / (crop[3] - crop[1]) * side)


def _arrow(ax, p, q, col, crop, ls="-", lw=1.7, ms=9):
    ax.add_patch(FancyArrowPatch(_pt(p, crop), _pt(q, crop), arrowstyle="-|>", mutation_scale=ms, color=col,
                                 lw=lw, linestyle=ls, shrinkA=0, shrinkB=0, zorder=5))


def _mark(ax, ok):
    if ok:
        ax.plot([0.72, 0.81, 0.94], [0.17, 0.07, 0.29], transform=ax.transAxes, color=OK, lw=2.4,
                solid_capstyle="round", zorder=6)
    else:
        for a, b in (([0.75, 0.94], [0.07, 0.26]), ([0.75, 0.94], [0.26, 0.07])):
            ax.plot(a, b, transform=ax.transAxes, color=BAD, lw=2.4, solid_capstyle="round", zorder=6)


def _tag(ax, text):
    ax.text(0.04, 0.96, text, transform=ax.transAxes, fontsize=6.0, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.8, pad=0.9, edgecolor="none"))


def panel_a(fig, gs):
    fr = _frames(DONOR)[: T_GRASP + 1]
    T = len(fr) - 1
    u = np.linspace(0, 1, N_STRIP)
    outer = gs.subgridspec(1, 2, width_ratios=[4.0, 1.25], wspace=0.05)
    rows = outer[0].subgridspec(2, 1, hspace=0.16)
    for r, (lab, col, w) in enumerate([("fast $\\rightarrow$ slow", FAST, 1 - (1 - u) ** 2),
                                       ("slow $\\rightarrow$ fast", SLOW, u ** 2)]):
        idx = np.round(w * T).astype(int)
        g = rows[r].subgridspec(2, N_STRIP, height_ratios=[1, 0.13], wspace=0.05, hspace=0.04)
        for c, i in enumerate(idx):
            ax = fig.add_subplot(g[0, c])
            end = c in (0, N_STRIP - 1)
            _img(ax, fr[i], CROP_A, edge=INK if end else col, lw=1.2 if end else 1.0)
            _tag(ax, f"$t{{=}}{i}$")
            if c == 0:
                ax.text(-0.09, 0.5, lab, transform=ax.transAxes, rotation=90, va="center", ha="center",
                        fontsize=7.4, color=col, fontweight="bold")
        tl = fig.add_subplot(g[1, :])                              # which source frames were sampled
        tl.plot([0, T], [0, 0], color=GRID, lw=1.2)
        tl.scatter(idx, np.zeros_like(idx), s=13, color=col, zorder=3, linewidths=0)
        tl.set_xlim(-1.5, T + 1.5); tl.set_ylim(-1, 1); tl.axis("off")

    # what each encoder returns: one z for both strips from the endpoints, two from the full transition
    z = fig.add_subplot(outer[1]); z.axis("off"); z.set_xlim(0, 20); z.set_ylim(0, 100)
    box = dict(boxstyle="round,pad=0,rounding_size=1.2", linewidth=0.9)
    y1, y2 = 75, 27                                                 # heights of the two strips' centres
    z.text(5.1, 60, "endpoints", fontsize=6.3, ha="center", va="bottom", color=INK2)      # each label sits on
    z.text(15.3, 84, "all frames", fontsize=6.3, ha="center", va="bottom", color=INK2)    # its own encoder
    for y, col in ((y1, FAST), (y2, SLOW)):
        z.add_patch(FancyArrowPatch((0.0, y), (1.6, 51 + (4 if y > 50 else -4)), arrowstyle="-", color=col,
                                    lw=0.9, connectionstyle="arc3,rad=0"))
    z.add_patch(FancyBboxPatch((1.6, 44), 7.0, 14, facecolor="white", edgecolor=BAD, **box))
    z.text(5.1, 51, "$z$", fontsize=8.6, ha="center", va="center", color=INK)
    z.text(5.1, 38, "same", fontsize=6.6, ha="center", va="top", color=BAD)
    for y, col, lab in ((y1, FAST, "$z_{\\mathrm{fast}}$"), (y2, SLOW, "$z_{\\mathrm{slow}}$")):
        z.add_patch(FancyBboxPatch((11.2, y - 7), 8.2, 14, facecolor="white", edgecolor=col, **box))
        z.text(15.3, y, lab, fontsize=8.2, ha="center", va="center", color=col)
    z.text(15.3, 51, "distinct", fontsize=6.6, ha="center", va="center", color=OK)


def panel_b(fig, gs):
    cells = gs.subgridspec(3, 4, width_ratios=[1, 0.13, 1, 1], height_ratios=[0.12, 1, 1], wspace=0.05,
                           hspace=0.08)
    for j, lab in ((0, "donor"), (2, "similar state"), (3, "shifted state")):
        t = fig.add_subplot(cells[0, j]); t.axis("off")
        t.text(0.5, 0.15, lab, fontsize=7.2, ha="center", va="center", color=INK if j == 0 else INK2)
    d = fig.add_subplot(cells[1:, 0]); d.axis("off")                # the donor, centred across both rows
    a = d.inset_axes([0.0, 0.26, 1.0, 0.48])
    _img(a, _frames(DONOR)[0], CROP_B, edge=INK2, lw=0.8)
    _arrow(a, GRIP, BOWL[DONOR], INK2, CROP_B, ls=(0, (3, 2)), lw=1.5)
    a.text(0.5, -0.06, "source effect", transform=a.transAxes, fontsize=6.4, color=INK2, ha="center", va="top")
    for r, (name, col) in enumerate((("recon. only", RECON), ("EAT (ours)", OURS))):
        lax = fig.add_subplot(cells[r + 1, 1]); lax.axis("off")
        lax.text(0.5, 0.5, name, rotation=90, va="center", ha="center", fontsize=7.2, color=col,
                 fontweight="bold", transform=lax.transAxes)
        for j, tag in ((2, SIMILAR), (3, SHIFTED)):
            ax = fig.add_subplot(cells[r + 1, j])
            fail = (r == 0 and tag == SHIFTED)
            _img(ax, _frames(tag)[0], CROP_B, edge=BAD if fail else OK, lw=1.3)
            _arrow(ax, GRIP, BOWL[DONOR], GRID, CROP_B, ls=(0, (3, 2)), lw=1.3, ms=7)   # source effect, as is
            target = BOWL[DONOR] if r == 0 else BOWL[tag]                                # absolute vs adapted
            _arrow(ax, GRIP, target, col, CROP_B)
            if fail:
                ax.add_patch(Circle(_pt(BOWL[DONOR], CROP_B), 17, fill=False, edgecolor=BAD, lw=1.1,
                                    ls=(0, (2, 1.5)), zorder=6))
            _mark(ax, not fail)


def teaser(out):
    fig = plt.figure(figsize=(7.2, 2.05))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.10)
    panel_a(fig, gs[0])
    panel_b(fig, gs[1])
    fig.text(0.01, 1.0, "(a)  Same endpoints, different dynamics", fontsize=8.8, fontweight="bold", va="bottom")
    fig.text(0.565, 1.0, "(b)  Same action, different states", fontsize=8.8, fontweight="bold", va="bottom")
    fig.savefig(out + ".pdf", pad_inches=0.05); fig.savefig(out + ".png", pad_inches=0.05); plt.close(fig)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    teaser(os.path.join("figures", "teaser"))
    print("wrote figures/teaser.{pdf,png}")
