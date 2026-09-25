"""Teaser: figures/teaser.{pdf,png} -- the two counter-examples, full width, page 1.

    ../envs/univla_vla/bin/python -m acdc_la.figures.make_teaser_figure

Visual language after Jenni et al. (ECCV 2020), SpeedNet (CVPR 2020) and MoCoGAN (CVPR 2018): show the
data first, explain the representation after.

(a) Same endpoints, different dynamics.  ONE real LIBERO-Spatial demonstration (approach phase, t = 0..61)
    resampled under two temporal warps, w(u) = 1-(1-u)^2 (fast -> slow) and w(u) = u^2 (slow -> fast).
    First and last frames are the same pixels by construction, so an encoder that reads only the endpoints
    receives identical inputs and must return identical latents -- a fact, not an illustration.
(b) Same action, different states.  Real LIBERO-Spatial scenes: the donor picks the black bowl at the table
    centre; the similar recipient is another demonstration of that task, the shifted recipient a scene whose
    bowl sits by the cookie box.  "After" frames are demonstration frames at the grasp.  The arrows are
    SCHEMATIC effect directions drawn on those real frames, not decoder outputs.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

INK, INK2, GRID = "#1a1a19", "#52514e", "#d9d9d6"
FAST, SLOW = "#2a78d6", "#eb6834"
OURS, BAD, OK = "#4a3aa7", "#d92d20", "#1baf7a"
RECON = "#eb6834"

EPS = "figures/teaser_eps/keep"
DONOR, SIMILAR, SHIFTED = "table_center__libero_spatial_0", "table_center__libero_spatial_1", \
                          "next_to_the_cookie_box__libero_spatial_0"
T_GRASP = {DONOR: 61, SIMILAR: 45, SHIFTED: 50}          # the grasp frame of each demonstration (read by eye)
GRIP = (0.50, 0.27)                                       # gripper at t = 0, frame fractions from top-left
BOWL = {DONOR: (0.48, 0.62), SIMILAR: (0.48, 0.62), SHIFTED: (0.62, 0.83)}
N_STRIP = 6
CROP = (0.12, 0.12, 0.88, 0.88)                            # x0, y0, x1, y1 in frame fractions, applied everywhere

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.4, "figure.dpi": 160, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "text.color": INK,
})


def _frames(tag):
    return np.load(os.path.join(EPS, tag + ".npy"))


def _crop(im):
    H, W = im.shape[:2]
    return im[int(CROP[1] * H):int(CROP[3] * H), int(CROP[0] * W):int(CROP[2] * W)]


def _pt(p, H):
    """Frame-fraction point -> pixel coordinates inside the crop."""
    return ((p[0] - CROP[0]) / (CROP[2] - CROP[0]) * H, (p[1] - CROP[1]) / (CROP[3] - CROP[1]) * H)


def _img(ax, im, edge=GRID, lw=0.6):
    ax.imshow(_crop(im)); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(edge); s.set_linewidth(lw)


def _arrow(ax, p, q, col, ls="-", lw=1.5, ms=8):
    H = int(256 * (CROP[2] - CROP[0]))
    ax.add_patch(FancyArrowPatch(_pt(p, H), _pt(q, H), arrowstyle="-|>", mutation_scale=ms,
                                 color=col, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0, zorder=5))


def _mark(ax, ok):
    if ok:
        ax.plot([0.74, 0.82, 0.94], [0.16, 0.07, 0.27], transform=ax.transAxes, color=OK, lw=2.0,
                solid_capstyle="round", zorder=6)
    else:
        for a, b in (([0.76, 0.93], [0.07, 0.24]), ([0.76, 0.93], [0.24, 0.07])):
            ax.plot(a, b, transform=ax.transAxes, color=BAD, lw=2.0, solid_capstyle="round", zorder=6)


def panel_a(fig, gs):
    fr = _frames(DONOR)[: T_GRASP[DONOR] + 1]
    T = len(fr) - 1
    u = np.linspace(0, 1, N_STRIP)
    warps = [("fast $\\rightarrow$ slow", FAST, 1 - (1 - u) ** 2), ("slow $\\rightarrow$ fast", SLOW, u ** 2)]
    inner = gs.subgridspec(2, 1, hspace=0.18)
    for r, (lab, col, w) in enumerate(warps):
        idx = np.round(w * T).astype(int)
        g = inner[r].subgridspec(2, N_STRIP, height_ratios=[1, 0.16], wspace=0.04, hspace=0.05)
        for c, i in enumerate(idx):
            ax = fig.add_subplot(g[0, c])
            end = c in (0, N_STRIP - 1)
            _img(ax, fr[i], edge=INK if end else col, lw=1.1 if end else 0.8)
            ax.text(0.04, 0.96, f"$t{{=}}{i}$", transform=ax.transAxes, fontsize=5.8, va="top", ha="left",
                    bbox=dict(facecolor="white", alpha=0.75, pad=0.8, edgecolor="none"))
            if c == 0:
                ax.text(-0.10, 0.5, lab, transform=ax.transAxes, rotation=90, va="center", ha="center",
                        fontsize=7.2, color=col, fontweight="bold")
        tl = fig.add_subplot(g[1, :])                            # source timeline: WHICH frames were sampled
        tl.plot([0, T], [0, 0], color=GRID, lw=1.2)
        tl.scatter(idx, np.zeros_like(idx), s=11, color=col, zorder=3, linewidths=0)
        tl.set_xlim(-1, T + 1); tl.set_ylim(-1, 1); tl.axis("off")
    return idx


def panel_a_latent(ax):
    """Endpoint encoding vs full transition, as a diagram of inputs rather than a latent-space cartoon."""
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    box = dict(boxstyle="round,pad=0,rounding_size=1.8", linewidth=0.7)
    ax.text(50, 97, "Endpoints only", fontsize=7.6, ha="center", va="top", color=INK, fontweight="bold")
    ax.text(50, 88, "reads $(x_0, x_T)$", fontsize=6.8, ha="center", va="top", color=INK2)
    for x, col in ((22, FAST), (78, SLOW)):
        ax.add_patch(FancyArrowPatch((x, 80), (46 if x < 50 else 54, 66), arrowstyle="-|>", mutation_scale=7,
                                     color=col, lw=1.1))
    ax.add_patch(FancyBboxPatch((38, 55), 24, 11, facecolor="white", edgecolor=INK2, **box))
    ax.text(50, 60.5, "$z$", fontsize=8, ha="center", va="center")
    ax.text(50, 50, "identical input $\\Rightarrow$ same $z$", fontsize=6.6, ha="center", va="top", color=BAD)

    ax.text(50, 38, "Full transition", fontsize=7.6, ha="center", va="top", color=INK, fontweight="bold")
    ax.text(50, 29, "reads every frame", fontsize=6.8, ha="center", va="top", color=INK2)
    for x, col, zx in ((22, FAST, 12), (78, SLOW, 64)):
        ax.add_patch(FancyArrowPatch((x, 22), (zx + 12, 12), arrowstyle="-|>", mutation_scale=7, color=col, lw=1.1))
        ax.add_patch(FancyBboxPatch((zx, 0), 24, 11, facecolor="white", edgecolor=col, **box))
        ax.text(zx + 12, 5.5, "$z_{\\mathrm{fast}}$" if col == FAST else "$z_{\\mathrm{slow}}$", fontsize=7.4,
                ha="center", va="center", color=col)
    ax.text(50, -5, "distinct latents", fontsize=6.6, ha="center", va="top", color=OK)


def panel_b(fig, gs):
    """MoCoGAN-style grid: rows = objective, columns = recipient state; each cell is before -> after."""
    cells = gs.subgridspec(3, 4, width_ratios=[0.5, 0.11, 1, 1], height_ratios=[0.14, 1, 1], wspace=0.06,
                           hspace=0.10)
    donor = _frames(DONOR)
    # donor column: the transition whose effect is transported
    axd = fig.add_subplot(cells[1:, 0])
    axd.axis("off")
    for k, (t, y) in enumerate(((0, 0.53), (T_GRASP[DONOR], 0.03))):
        a = axd.inset_axes([0.0, y, 1.0, 0.44])
        _img(a, donor[t], edge=INK2, lw=0.7)
        if k == 0: _arrow(a, GRIP, BOWL[DONOR], INK2, ls=(0, (3, 2)), lw=1.3)
        a.text(0.04, 0.96, "before" if k == 0 else "after", transform=a.transAxes, fontsize=5.8, va="top",
               bbox=dict(facecolor="white", alpha=0.75, pad=0.8, edgecolor="none"))
    for j, lab in ((0, "donor"), (2, "similar state"), (3, "shifted state")):
        t = fig.add_subplot(cells[0, j]); t.axis("off")
        t.text(0.5, 0.2, lab, fontsize=7.2, ha="center", va="center", color=INK if lab == "donor" else INK2)
    for r, (name, col) in enumerate((("recon. only", RECON), ("EAT (ours)", OURS))):
        lax = fig.add_subplot(cells[r + 1, 1]); lax.axis("off")
        lax.text(0.5, 0.5, name, rotation=90, va="center", ha="center", fontsize=7.0, color=col,
                 fontweight="bold", linespacing=1.0, transform=lax.transAxes)
        for j, tag in ((2, SIMILAR), (3, SHIFTED)):
            fr = _frames(tag)
            g = cells[r + 1, j].subgridspec(1, 2, wspace=0.03)
            before, after = fig.add_subplot(g[0]), fig.add_subplot(g[1])
            _img(before, fr[0])
            _arrow(before, GRIP, BOWL[DONOR], GRID, ls=(0, (3, 2)), lw=1.1, ms=6)      # the donor's absolute effect
            fail = (r == 0 and tag == SHIFTED)
            target = BOWL[DONOR] if r == 0 else BOWL[tag]                               # recon: absolute; EAT: relative
            _arrow(before, GRIP, target, col, lw=1.6)
            if fail:                                                                     # where it lands: empty table
                Hc = int(256 * (CROP[2] - CROP[0]))
                before.add_patch(Circle(_pt(BOWL[DONOR], Hc), 15, fill=False, edgecolor=BAD, lw=1.0,
                                        ls=(0, (2, 1.5)), zorder=6))
                _img(after, fr[0], edge=BAD, lw=1.4)
                after.add_patch(Rectangle((0, 0), 1, 1, transform=after.transAxes, color="white", alpha=0.6, zorder=4))
                after.text(0.5, 0.58, "empty\ntable", transform=after.transAxes, ha="center", va="center",
                           fontsize=6.6, color=BAD, zorder=6, linespacing=1.1)
                _mark(after, False)
            else:
                _img(after, fr[T_GRASP[tag]], edge=OK, lw=1.4)
                _mark(after, True)


def teaser(out):
    fig = plt.figure(figsize=(7.2, 2.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.85, 0.62, 2.05], wspace=0.10)
    panel_a(fig, gs[0])
    panel_a_latent(fig.add_subplot(gs[1]))
    panel_b(fig, gs[2])
    fig.text(0.01, 1.03, "(a)  Same endpoints, different dynamics", fontsize=8.8, fontweight="bold", va="bottom")
    fig.text(0.01, 0.975, "A latent action should preserve how the transition unfolds.", fontsize=7.0,
             color=INK2, va="bottom", style="italic")
    fig.text(0.545, 1.03, "(b)  Same action, different states", fontsize=8.8, fontweight="bold", va="bottom")
    fig.text(0.545, 0.975, "Its effect should adapt to the current state.", fontsize=7.0, color=INK2,
             va="bottom", style="italic")
    fig.text(0.5, -0.045, "Latent actions should capture temporal effects — not merely endpoints — "
             "and remain executable across states.", ha="center", fontsize=7.6, color=INK, fontweight="bold")
    fig.savefig(out + ".pdf", pad_inches=0.06); fig.savefig(out + ".png", pad_inches=0.06); plt.close(fig)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(root)
    teaser(os.path.join("figures", "teaser"))
    print("wrote figures/teaser.{pdf,png}")
