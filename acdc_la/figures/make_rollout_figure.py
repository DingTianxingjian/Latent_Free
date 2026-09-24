"""Qualitative rollout comparison: figures/rollout_compare.{pdf,png}.

    PYTHONPATH=. ../envs/univla_vla/bin/python -m acdc_la.figures.make_rollout_figure

Two tasks side by side.  Within a task both rows start from the SAME episode and see the SAME nuisance
intervention, so the only thing that differs is the arm: UniVLA* above, TERRA below.  The failure step
carries a red frame, a red cross, a red box around the object the failure is about and a magnified crop
of it; the completed step carries the same in green.  Step numbers run along the top of each task.

MOCK = True  ->  frames are demonstration observations from data/libero_rlds, not policy rollouts, and the
figure prints a footer saying so.  The layout is final: when the two arms have been rolled out from one
initial state and one seed, write their frames into figures/rollout_frames/ as <tag>_<0..7>.png, point the
rows at those tags, set the ROIs to where the failure actually is, and set MOCK = False.
"""
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image

from ..data.distractor import Distractor
from ..paths import IMG

MOCK = True
INK, INK2, GRID = "#1a1a19", "#52514e", "#d9d9d6"
OURS, BASE = "#4a3aa7", "#2a78d6"          # TERRA violet, UniVLA blue -- the same hues as the probe figures
OK, BAD = "#1baf7a", "#d92d20"
EP = 7                                     # episode id: seeds the patch trajectory

# Rows of one task.  `idx` indexes the 8 frames extracted per clip; `roi` is (x0, y0, x1, y1) in frame
# fractions, the region the reader is meant to look at on the LAST frame.  [MOCK] the ROIs below are
# placed by eye on demonstration frames; with real rollouts they go where the failure actually is.
BLOCKS = [
    {"tag": "libero_object", "title": "LIBERO-Object", "perturb": "distractor",
     "nuisance": "moving-patch distractor (dashed box)",
     "base": {"idx": [0, 2, 3, 4], "note": "grasps the wrong object", "roi": (0.40, 0.34, 0.68, 0.66)},
     "ours": {"idx": [0, 3, 5, 7], "roi": (0.06, 0.34, 0.38, 0.72)}},
    {"tag": "libero_10", "title": "LIBERO-Long", "perturb": "lighting",
     "nuisance": "directional lighting (0.55\u20131.25$\\times$ across the frame)",
     "base": {"idx": [0, 2, 3, 4], "note": "releases before the second stage", "roi": (0.16, 0.44, 0.46, 0.80)},
     "ours": {"idx": [0, 3, 5, 7], "roi": (0.52, 0.44, 0.86, 0.82)}},
]
ROWS = [("UniVLA$^*$", BASE, "base"), ("TERRA", OURS, "ours")]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.4, "figure.dpi": 160, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "text.color": INK,
})


def _clip(d, tag, n=8):
    x = np.stack([np.asarray(Image.open(os.path.join(d, f"{tag}_{i}.png")).resize((IMG, IMG))) for i in range(n)])
    return torch.from_numpy(x).permute(0, 3, 1, 2).float().div_(255.)[None]


def _perturb(clip, kind, dis):
    if kind == "distractor":
        return dis.render(clip, torch.tensor([EP]))
    if kind == "lighting":
        # A DIRECTIONAL illumination gradient, 0.55x on one side to 1.25x on the other.  Deliberately NOT the
        # invariance term's family: that is a global brightness/contrast scalar (U(0.8,1.2)), so testing with it
        # would put the perturbation inside our own training distribution and outside the baseline's.
        w = torch.linspace(0.55, 1.25, clip.shape[-1]).view(1, 1, 1, 1, -1)
        return (clip * w).clamp(0, 1)
    return clip


def _show(ax, img, edge=GRID, lw=0.6):
    ax.imshow(img.permute(1, 2, 0).numpy())
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(edge); s.set_linewidth(lw)


def _mark(ax, img, roi, col):
    """Box the region the reader should look at, and magnify it into the frame's top-right corner."""
    h = img.shape[-2]
    x0, y0, x1, y1 = roi
    ax.add_patch(Rectangle((x0 * h, y0 * h), (x1 - x0) * h, (y1 - y0) * h, fill=False, edgecolor=col,
                           linewidth=1.0, zorder=5))
    crop = img[:, int(y0 * h):int(y1 * h), int(x0 * h):int(x1 * h)]
    ins = ax.inset_axes([0.58, 0.58, 0.42, 0.42])
    ins.imshow(crop.permute(1, 2, 0).numpy())
    ins.set_xticks([]); ins.set_yticks([])
    for s in ins.spines.values():
        s.set_color(col); s.set_linewidth(1.1)


def _check(ax):
    ax.plot([0.80, 0.865, 0.95], [0.135, 0.07, 0.22], transform=ax.transAxes, color=OK, lw=2.0,
            solid_capstyle="round", clip_on=False, zorder=6)


def _cross(ax):
    for a, b in (([0.81, 0.94], [0.07, 0.20]), ([0.81, 0.94], [0.20, 0.07])):
        ax.plot(a, b, transform=ax.transAxes, color=BAD, lw=2.0, solid_capstyle="round", clip_on=False, zorder=6)


def rollout_compare(out):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    d = os.path.join(root, "figures", "rollout_frames")
    meta = {m["suite"]: m for m in json.load(open(os.path.join(d, "frames.json")))}
    nf = len(BLOCKS[0]["base"]["idx"])
    dis = Distractor("cpu")
    fig = plt.figure(figsize=(7.2, 2.25))
    outer = fig.add_gridspec(1, 2, wspace=0.085)
    for blk in BLOCKS:
        clip = _perturb(_clip(d, blk["tag"]), blk["perturb"], dis)      # ONE clip: both rows share the state
        steps = np.linspace(0, meta[blk["tag"]]["episode_len"] - 1, 8).round().astype(int)
        gs = outer[BLOCKS.index(blk)].subgridspec(2, nf, wspace=0.035, hspace=0.13)
        axs = [[None] * nf, [None] * nf]
        for ri, (name, col, key) in enumerate(ROWS):
            spec = blk[key]
            for ci, fi in enumerate(spec["idx"]):
                ax = fig.add_subplot(gs[ri, ci]); axs[ri][ci] = ax
                last = ci == nf - 1
                edge, lw = (BAD, 1.6) if (last and key == "base") else ((OK, 1.6) if last else (GRID, 0.6))
                _show(ax, clip[0, fi], edge, lw)
                if ci == 0 and ri == 0 and blk["perturb"] == "distractor":     # name the patch: it reads as a
                    ax.add_patch(Rectangle((3, IMG - 67), 64, 64, fill=False, edgecolor=INK2,  # camera window otherwise
                                           linewidth=0.8, linestyle=(0, (2, 1.6)), zorder=6))
                    ax.text(35, IMG - 72, "distractor", fontsize=5.8, color=INK2, ha="center", va="bottom", zorder=6)
                if last:
                    _mark(ax, clip[0, fi], spec["roi"], BAD if key == "base" else OK)
                    (_cross if key == "base" else _check)(ax)
                if ci == 0:
                    ax.add_patch(Rectangle((-0.085, 0), 0.05, 1, transform=ax.transAxes, color=col,
                                           clip_on=False, zorder=5))
                    ax.text(-0.14, 0.5, name, transform=ax.transAxes, rotation=90, va="center", ha="center",
                            fontsize=7.0, color=col, fontweight="bold", zorder=6)
                ax.text(0.035, 0.965, f"$t{{=}}{steps[fi]}$", transform=ax.transAxes, fontsize=5.8,
                        color=INK, va="top", ha="left", zorder=6,       # per FRAME: the two rows are sampled at
                        bbox=dict(facecolor="white", alpha=0.72, pad=1.0, edgecolor="none"))   # different steps
                if last and key == "base":
                    ax.text(0.5, -0.045, spec["note"], transform=ax.transAxes, fontsize=6.8, color=BAD,
                            ha="center", va="top")
        # task header: title, full instruction, and which nuisance both rows are under
        bb = [axs[0][c].get_position() for c in range(nf)]
        x0, x1 = bb[0].x0, bb[-1].x1
        ins = textwrap.wrap(meta[blk["tag"]]["instruction"], width=46)     # the header is three short lines, so a
        fig.text(x0, bb[0].y1 + 0.125, blk["title"], fontsize=8.4,         # task's text never reaches its neighbour
                 fontweight="bold", va="bottom", color=INK)
        fig.text(x0, bb[0].y1 + 0.075, "“" + "\n".join(ins) + "”", fontsize=6.6, va="bottom",
                 color=INK2, style="italic", linespacing=1.25)
        fig.text(x0, bb[0].y1 + 0.028, "both rows under the " + blk["nuisance"], fontsize=6.4,
                 va="bottom", color=INK2)
        y = axs[1][0].get_position().y0 - 0.032                                # a thin time axis under the task
        fig.add_artist(plt.Line2D([x0, x1], [y, y], color=GRID, lw=0.8))
        fig.add_artist(plt.matplotlib.patches.FancyArrowPatch((x1 - 0.012, y), (x1, y), arrowstyle="-|>",
                                                             mutation_scale=7, color=GRID, lw=0.8))
        fig.text(x1, y - 0.035, "time", fontsize=6.3, color=INK2, ha="right", va="top", style="italic")
    if MOCK:
        fig.text(0.5, -0.055, "layout mock-up: frames are demonstration observations under the paper's own "
                 "nuisance interventions, not policy rollouts", ha="center", fontsize=6.6, color=BAD, style="italic")
    fig.savefig(out + ".pdf", pad_inches=0.06); fig.savefig(out + ".png", pad_inches=0.06); plt.close(fig)


if __name__ == "__main__":
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "figures")
    rollout_compare(os.path.join(d, "rollout_compare"))
    print("wrote", d + "/rollout_compare.{pdf,png}")
