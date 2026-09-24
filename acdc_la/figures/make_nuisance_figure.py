"""What the robustness claim looks like: figures/nuisance_examples.{pdf,png}.

    ../envs/univla_vla/bin/python -m acdc_la.figures.make_nuisance_figure

Real LIBERO observations under the two perturbations this project actually measures, rendered by the
project's own code, not by an illustration:
  row 2  the moving-patch distractor of acdc_la/data/distractor.py -- a 64x64 natural-video crop
         travelling around the frame border, the intervention behind every "drop" number in the paper
  row 3  the brightness / contrast jitter of the invariance term (stage3_transport/train_v2.py::_jitter),
         one draw shared by every frame of a clip
The bar panel is measured: relative loss of Net R^2 under the distractor (LEDGER 2026-09-11 / 09-14 / 09-16).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image

from ..data.distractor import Distractor
from ..paths import IMG

INK, INK2, GRID = "#1a1a19", "#52514e", "#d9d9d6"
OURS, UNIVLA, POOLED = "#4a3aa7", "#2a78d6", "#a294db"
# The measured drops (endpoint-only 33.6 %, UniVLA 22.9 %, ours 17.9 %) belong in the caption, not on the figure:
# this one shows what the intervention IS; Figure~ref{fig:representation_comparison}(c) already plots the numbers.
ROWS = [("clean", "#b5b5b1"), ("distractor", INK2), ("lighting", INK2)]
BLOCKS = [("libero_object", "LIBERO-Object"), ("libero_10", "LIBERO-Long")]
EP = 7                       # episode id: seeds the patch trajectory, so the patch moves coherently

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 7.4, "figure.dpi": 160, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "text.color": INK,
})


def _load(d, suite, idx):
    x = np.stack([np.asarray(Image.open(os.path.join(d, f"{suite}_{i}.png")).resize((IMG, IMG))) for i in idx])
    return torch.from_numpy(x).permute(0, 3, 1, 2).float().div_(255.)[None]      # (1,T,3,IMG,IMG)


def _jitter_fixed(frames, bright=0.80, contrast=1.20):
    """The invariance term's photometric family at a fixed draw, so the figure is reproducible."""
    x = frames * bright
    return ((x - x.mean(dim=(-1, -2), keepdim=True)) * contrast + x.mean(dim=(-1, -2), keepdim=True)).clamp(0, 1)


def _show(ax, img, edge=GRID, lw=0.6):
    ax.imshow(img.permute(1, 2, 0).numpy())
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(edge); s.set_linewidth(lw)


def nuisance(out):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    d = os.path.join(root, "figures", "rollout_frames")
    keep = [0, 3, 5, 7]
    dis = Distractor("cpu")
    fig = plt.figure(figsize=(7.2, 2.9))
    outer = fig.add_gridspec(1, 2, wspace=0.13)
    for bi, (suite, title) in enumerate(BLOCKS):
        full = _load(d, suite, range(8))                       # render on the whole clip: the patch trajectory is
        views = [full[:, keep],                                # a function of the frame index, so it travels
                 dis.render(full, torch.tensor([EP]))[:, keep],
                 _jitter_fixed(full[:, keep])]
        gs = outer[bi].subgridspec(3, len(keep), wspace=0.04, hspace=0.05)
        for ri, batch in enumerate(views):
            name, col = ROWS[ri]
            for ci in range(len(keep)):
                ax = fig.add_subplot(gs[ri, ci])
                _show(ax, batch[0, ci])
                if ci == 0:
                    ax.add_patch(Rectangle((-0.085, 0), 0.05, 1, transform=ax.transAxes, color=col,
                                           clip_on=False, zorder=5))
                    ax.text(-0.17, 0.5, name, transform=ax.transAxes, rotation=90, va="center", ha="center",
                            fontsize=7.0, color=INK if ri == 0 else INK2, zorder=6,
                            fontweight="normal" if ri == 0 else "bold")
                if ri == 0 and ci == 0:
                    ax.set_title(title, loc="left", fontsize=8.4, fontweight="bold", pad=3.5, color=INK)
    fig.savefig(out + ".pdf", pad_inches=0.08); fig.savefig(out + ".png", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "figures")
    nuisance(os.path.join(d, "nuisance_examples"))
    print("wrote", d + "/nuisance_examples.{pdf,png}")
