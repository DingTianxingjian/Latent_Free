"""Qualitative rollout comparison: figures/rollout_compare.{pdf,png}.

    ../envs/univla_vla/bin/python -m acdc_la.figures.make_rollout_figure

Two tasks side by side; each task is two rows from the same initial state, UniVLA* above TERRA.
The failure step carries a red frame and a short label, the completed step a green frame and a check.

MOCK = True  ->  the frames come from a demonstration in figures/rollout_frames/, not from a policy
rollout, and the figure says so in a footer.  When the real evaluations are in, drop the frames of the
two arms into the same directory, set MOCK = False and keep the layout.
"""
import json
import os

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
OURS, BASE = "#4a3aa7", "#eb6834"
OK, BAD = "#1baf7a", "#d92d20"

# Which frames each row shows, which clip they come from, and what the failure is meant to illustrate.
# `perturb`: the nuisance applied to BOTH rows of the block, so the two arms see the same scene.
#   "distractor" = the moving patch of acdc_la/data/distractor.py (the intervention the paper measures)
#   "lighting"   = one draw of the invariance term's brightness/contrast family
BLOCKS = [
    {"suite": "libero_object", "title": "LIBERO-Object", "perturb": "distractor",
     "base": {"suite": "libero_object_alt", "idx": [0, 3, 5, 7], "note": "grasps the wrong object"},
     "ours": {"suite": "libero_object", "idx": [0, 3, 5, 7]}},
    {"suite": "libero_10", "title": "LIBERO-Long", "perturb": "lighting",
     "base": {"suite": "libero_10", "idx": [0, 2, 3, 4], "note": "releases before the second stage"},
     "ours": {"suite": "libero_10", "idx": [0, 3, 5, 7]}},
]
EP = 7                    # episode id: seeds the patch trajectory, so the patch travels coherently
ROWS = [("UniVLA$^*$", BASE, "base"), ("TERRA", OURS, "ours")]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 7.4, "figure.dpi": 160, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "text.color": INK,
})


def _clip(d, suite, n=8):
    x = np.stack([np.asarray(Image.open(os.path.join(d, f"{suite}_{i}.png")).resize((IMG, IMG))) for i in range(n)])
    return torch.from_numpy(x).permute(0, 3, 1, 2).float().div_(255.)[None]


def _perturb(clip, kind, dis):
    if kind == "distractor":
        return dis.render(clip, torch.tensor([EP]))
    if kind == "lighting":                                   # brightness 0.80 x contrast 1.20, one draw per clip
        x = clip * 0.80
        return ((x - x.mean(dim=(-1, -2), keepdim=True)) * 1.20 + x.mean(dim=(-1, -2), keepdim=True)).clamp(0, 1)
    return clip


def _frame(ax, img, edge=GRID, lw=0.6):
    ax.imshow(img.permute(1, 2, 0).numpy())
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(edge); s.set_linewidth(lw)


def rollout_compare(out):
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "figures", "rollout_frames")
    meta = {m["suite"]: m for m in json.load(open(os.path.join(d, "frames.json")))}
    nf = len(BLOCKS[0]["base"]["idx"])
    dis = Distractor("cpu")
    fig = plt.figure(figsize=(7.2, 2.05))
    outer = fig.add_gridspec(1, 2, wspace=0.13)
    for bi, blk in enumerate(BLOCKS):
        gs = outer[bi].subgridspec(2, nf, wspace=0.04, hspace=0.30)
        for ri, (name, col, key) in enumerate(ROWS):
            spec = blk[key]
            clip = _perturb(_clip(d, spec["suite"]), blk["perturb"], dis)
            for ci, fi in enumerate(spec["idx"]):
                ax = fig.add_subplot(gs[ri, ci])
                last = ci == nf - 1
                edge, lw = GRID, 0.6
                if last: edge, lw = (BAD, 1.6) if key == "base" else (OK, 1.6)
                _frame(ax, clip[0, fi], edge, lw)
                if ci == 0:                                    # method bar + name on the left
                    ax.add_patch(Rectangle((-0.085, 0), 0.05, 1, transform=ax.transAxes, color=col,
                                           clip_on=False, zorder=5))
                    ax.text(-0.13, 0.5, name, transform=ax.transAxes, rotation=90, va="center", ha="center",
                            fontsize=7.0, color=col, fontweight="bold")
                if last and key == "ours":                     # drawn, not a glyph: DejaVu Serif has no U+2713
                    ax.plot([0.80, 0.865, 0.95], [0.135, 0.07, 0.22], transform=ax.transAxes,
                            color=OK, lw=2.0, solid_capstyle="round", clip_on=False, zorder=6)
                if last and key == "base":
                    ax.text(0.5, -0.12, spec["note"], transform=ax.transAxes, fontsize=6.6, color=BAD,
                            ha="center", va="top")
                if ri == 0 and ci == 0:
                    ax.set_title(blk["title"], loc="left", fontsize=8.4, fontweight="bold", pad=3.5, color=INK)
                if ri == 0 and ci == nf - 1:
                    ins = meta[blk["suite"]]["instruction"]
                    ins = ins if len(ins) <= 22 else ins[:22].rstrip() + "…"
                    ax.set_title(f"“{ins}”", loc="right", fontsize=6.4, color=INK2, pad=3.5, style="italic")
    if MOCK:
        fig.text(0.5, -0.005, "layout mock-up: frames are demonstration observations under the paper's own "
                 "nuisance interventions, not policy rollouts",
                 ha="center", fontsize=6.6, color=BAD, style="italic")
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png"); plt.close(fig)


if __name__ == "__main__":
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "figures")
    rollout_compare(os.path.join(d, "rollout_compare"))
    print("wrote", d + "/rollout_compare.{pdf,png}")
