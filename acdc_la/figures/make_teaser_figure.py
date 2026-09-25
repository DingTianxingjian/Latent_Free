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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Wedge

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

from matplotlib import font_manager as _fm
_FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
for _f in sorted(os.listdir(_FONTS)) if os.path.isdir(_FONTS) else []:
    if _f.endswith(".ttf"): _fm.fontManager.addfont(os.path.join(_FONTS, _f))
plt.rcParams.update({                                      # Comic Neue: an open (OFL) redrawing of Comic Sans
    "font.family": "Comic Neue", "mathtext.fontset": "custom", "mathtext.rm": "Comic Neue",
    "mathtext.it": "Comic Neue:italic", "mathtext.bf": "Comic Neue:bold",
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


def _space(ax, title, sub):
    """A latent space, drawn as a light rounded square with the clips' codes as coloured dots."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor="#f4f4f2", edgecolor=GRID, linewidth=0.8))
    ax.text(0.5, 1.21, title, fontsize=7.0, ha="center", va="bottom", color=INK, fontweight="bold")
    ax.text(0.5, 1.04, sub, fontsize=6.2, ha="center", va="bottom", color=INK2)


def panel_a(fig, gs):
    fr = _frames(DONOR)[: T_GRASP + 1]
    T = len(fr) - 1
    u = np.linspace(0, 1, N_STRIP)
    outer = gs.subgridspec(1, 2, width_ratios=[4.0, 1.15], wspace=0.30)
    rows = outer[0].subgridspec(2, 1, hspace=0.10)
    first, strips = None, []
    for r, (lab, col, w) in enumerate([("fast $\\rightarrow$ slow", FAST, 1 - (1 - u) ** 2),
                                       ("slow $\\rightarrow$ fast", SLOW, u ** 2)]):
        idx = np.round(w * T).astype(int)
        g = rows[r].subgridspec(3, N_STRIP, height_ratios=[0.20, 1, 0.13], wspace=0.05, hspace=0.04)
        head = fig.add_subplot(g[0, :]); head.axis("off")          # the row's name, horizontal, above its strip
        head.text(0.0, 0.1, lab, fontsize=8.0, color=col, fontweight="bold", ha="left", va="bottom",
                  transform=head.transAxes)
        for c, i in enumerate(idx):
            ax = fig.add_subplot(g[1, c])
            end = c in (0, N_STRIP - 1)
            _img(ax, fr[i], CROP_A, edge=INK if end else col, lw=1.2 if end else 1.0)
            _tag(ax, f"$t{{=}}{i}$")
            if first is None: first = ax
            if c == N_STRIP - 1: strips.append(ax)
        tl = fig.add_subplot(g[2, :])                              # which source frames were sampled
        tl.plot([0, T], [0, 0], color=GRID, lw=1.2)
        tl.scatter(idx, np.zeros_like(idx), s=13, color=col, zorder=3, linewidths=0)
        tl.set_xlim(-1.5, T + 1.5); tl.set_ylim(-1, 1); tl.axis("off")

    # what each encoder can recover: the progress curves the two strips were sampled from.  They are the
    # actual warps, so the panel is exact: identical at the endpoints, different everywhere in between.
    # Side by side and fed by one bracket over BOTH strips, so neither plot reads as belonging to one row.
    rc = outer[1].subgridspec(2, 1, hspace=0.95)
    uu = np.linspace(0, 1, 100)
    curves = ((FAST, 1 - (1 - uu) ** 2, 1 - (1 - u) ** 2), (SLOW, uu ** 2, u ** 2))
    plots = []
    for k, (title, verdict, vcol) in enumerate((("endpoints only", "same $z$", BAD),
                                                ("all frames", "distinct $z$", OK))):
        ax = fig.add_subplot(rc[k]); plots.append(ax)
        for col, c, samp in curves:
            ax.plot(uu, c, color=col, lw=1.6 if k else 1.1, alpha=1.0 if k else 0.28, zorder=2)
            if k:
                ax.scatter(u[1:-1], samp[1:-1], s=14, color=col, zorder=4, linewidths=0.6, edgecolors="white")
        ax.scatter([0, 1], [0, 1], s=22, color=INK, zorder=5, linewidths=0.6, edgecolors="white")
        if not k:
            ax.text(0.5, 0.5, "?", fontsize=10, ha="center", va="center", color=INK2, zorder=6)
        ax.set_xlim(-0.06, 1.06); ax.set_ylim(-0.06, 1.06); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sd in ("top", "right"): ax.spines[sd].set_visible(False)
        for sd in ("left", "bottom"): ax.spines[sd].set_color(GRID); ax.spines[sd].set_linewidth(0.8)
        ax.text(0.5, 1.08, title, transform=ax.transAxes, fontsize=6.8, ha="center", va="bottom",
                color=INK, fontweight="bold")
        ax.text(0.5, -0.10, verdict, transform=ax.transAxes, fontsize=7.0, ha="center", va="top", color=vcol,
                fontweight="bold")
    # the bracket: from the middle of each strip into the plots
    pa, pb = strips[0].get_position(), strips[1].get_position()
    ya, yb = (pa.y0 + pa.y1) / 2, (pb.y0 + pb.y1) / 2
    px = plots[0].get_position().x0
    x0 = pa.x1 + 0.004
    xb = pa.x1 + 0.25 * (px - pa.x1)
    for y in (ya, yb):
        fig.add_artist(plt.Line2D([x0, xb], [y, y], color=GRID, lw=1.0))
    fig.add_artist(plt.Line2D([xb, xb], [yb, ya], color=GRID, lw=1.0))
    ym = (ya + yb) / 2
    fig.add_artist(FancyArrowPatch((xb, ym), (px - 0.006, ym), arrowstyle="-|>",
                                   mutation_scale=7, color=INK2, lw=1.0, transform=fig.transFigure))
    return first


def panel_b(fig, gs):
    cells = gs.subgridspec(4, 5, width_ratios=[1, 0.34, 0.13, 1, 1], height_ratios=[0.12, 1, 1, 0.16],
                           wspace=0.05, hspace=0.08)
    for j, lab in ((0, "source clip"), (3, "similar state"), (4, "shifted state")):
        t = fig.add_subplot(cells[0, j]); t.axis("off")
        t.text(0.5, 0.15, lab, fontsize=7.2, ha="center", va="center", color=INK if j == 0 else INK2)
    d = fig.add_subplot(cells[1:3, 0]); d.axis("off")              # the clip whose latent is transported
    a = d.inset_axes([0.0, 0.27, 1.0, 0.46])
    _img(a, _frames(DONOR)[0], CROP_B, edge=INK2, lw=0.8)
    _arrow(a, GRIP, BOWL[DONOR], INK2, CROP_B, ls=(0, (3, 2)), lw=1.5)
    a.text(0.5, -0.05, "observed effect", transform=a.transAxes, fontsize=6.4, color=INK2, ha="center", va="top")
    zc = fig.add_subplot(cells[1:3, 1]); zc.axis("off"); zc.set_xlim(0, 1); zc.set_ylim(0, 1)
    zc.text(0.5, 0.5, "$z$", fontsize=8.6, ha="center", va="center", color=OURS, zorder=4, transform=zc.transAxes,
            bbox=dict(boxstyle="circle,pad=0.35", facecolor="#e7e3f5", edgecolor=OURS, linewidth=0.9))
    for y in (0.78, 0.22):                                          # the same z goes to both rows
        zc.add_patch(FancyArrowPatch((0.62, 0.5 + (0.07 if y > 0.5 else -0.07)), (0.98, y), arrowstyle="-|>",
                                     mutation_scale=7, color=INK2, lw=0.9, transform=zc.transAxes, clip_on=False))
    zc.add_patch(FancyArrowPatch((-0.20, 0.5), (0.30, 0.5), arrowstyle="-|>", mutation_scale=7, color=INK2,
                                 lw=0.9, transform=zc.transAxes, clip_on=False))
    for r, (name, col) in enumerate((("recon. only", RECON), ("EAT (ours)", OURS))):
        lax = fig.add_subplot(cells[r + 1, 2]); lax.axis("off")
        lax.text(0.5, 0.5, name, rotation=90, va="center", ha="center", fontsize=7.2, color=col,
                 fontweight="bold", transform=lax.transAxes)
        for j, tag in ((3, SIMILAR), (4, SHIFTED)):
            ax = fig.add_subplot(cells[r + 1, j])
            fail = (r == 0 and tag == SHIFTED)
            _img(ax, _frames(tag)[0], CROP_B, edge=BAD if fail else OK, lw=1.3)
            _arrow(ax, GRIP, BOWL[DONOR], GRID, CROP_B, ls=(0, (3, 2)), lw=1.3, ms=7)
            target = BOWL[DONOR] if r == 0 else BOWL[tag]
            _arrow(ax, GRIP, target, col, CROP_B)
            if fail:
                ax.add_patch(Circle(_pt(BOWL[DONOR], CROP_B), 17, fill=False, edgecolor=BAD, lw=1.1,
                                    ls=(0, (2, 1.5)), zorder=6))
            _mark(ax, not fail)
    lg = fig.add_subplot(cells[3, 3:]); lg.axis("off"); lg.set_xlim(0, 1); lg.set_ylim(0, 1)
    lg.add_patch(FancyArrowPatch((0.02, 0.35), (0.12, 0.35), arrowstyle="-|>", mutation_scale=6, color=GRID,
                                 lw=1.2, linestyle=(0, (3, 2))))
    lg.text(0.14, 0.35, "observed effect", fontsize=6.2, color=INK2, va="center")
    lg.add_patch(FancyArrowPatch((0.02, -0.35), (0.12, -0.35), arrowstyle="-|>", mutation_scale=6, color=OURS,
                                 lw=1.4, clip_on=False))
    lg.text(0.14, -0.35, "executed from $z$", fontsize=6.2, color=INK2, va="center")
    return a


def teaser(out):
    fig = plt.figure(figsize=(7.2, 2.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.07, top=0.90)
    fa = panel_a(fig, gs[0])
    fb = panel_b(fig, gs[1])
    top = 0.975                                                     # titles start where each panel's frames start
    fig.text(fa.get_position().x0, top, "(a)  Same endpoints, different dynamics", fontsize=8.8,
             fontweight="bold", va="bottom")
    fig.text(fb.get_position().x0, top, "(b)  Same action, different states", fontsize=8.8,
             fontweight="bold", va="bottom")
    fig.savefig(out + ".pdf", pad_inches=0.05); fig.savefig(out + ".png", pad_inches=0.05); plt.close(fig)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    teaser(os.path.join("figures", "teaser"))
    print("wrote figures/teaser.{pdf,png}")
