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
from PIL import Image

from ..data.distractor import Distractor
from ..paths import IMG

INK, INK2, GRID = "#1a1a19", "#52514e", "#d9d9d6"
OURS, UNIVLA, POOLED = "#4a3aa7", "#2a78d6", "#a294db"
DROP = {"endpoint only": 33.6, "UniVLA (matched)": 22.9, "TERRA (ours)": 17.9}   # [real] Net R^2 lost, %
ROWS = ["clean observation", "moving-patch distractor", "brightness / contrast"]
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


def _show(ax, img):
    ax.imshow(img.permute(1, 2, 0).numpy())
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(GRID); s.set_linewidth(0.6)


def nuisance(out):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    d = os.path.join(root, "figures", "rollout_frames")
    full = _load(d, "libero_object", range(8))          # render on the whole clip: the patch trajectory is a
    patched_full = Distractor("cpu").render(full, torch.tensor([EP]))   # function of the frame index, so the
    keep = [0, 3, 5, 7]                                                 # patch has visibly travelled by frame 7
    clean, patched = full[:, keep], patched_full[:, keep]
    jittered = _jitter_fixed(clean)
    nf = len(keep)

    fig = plt.figure(figsize=(7.2, 2.7))
    gs = fig.add_gridspec(3, nf + 1, width_ratios=[1] * nf + [1.35], wspace=0.10, hspace=0.10)
    for ri, batch in enumerate((clean, patched, jittered)):
        for ci in range(nf):
            ax = fig.add_subplot(gs[ri, ci])
            _show(ax, batch[0, ci])
            if ci == 0:
                ax.set_ylabel(ROWS[ri], fontsize=7.0, color=INK if ri == 0 else INK2, labelpad=3,
                              rotation=0, ha="right", va="center")
                ax.yaxis.set_label_coords(-0.06, 0.5)

    ax = fig.add_subplot(gs[:, nf])
    ks = list(DROP); v = [DROP[k] for k in ks]
    bars = ax.barh(range(len(ks))[::-1], v, height=0.38, color=[POOLED, UNIVLA, OURS],
                   edgecolor="white", linewidth=0.8, zorder=3)
    for b, vv, k in zip(bars, v, ks):                     # labels above each bar: no tick labels to collide
        ax.text(vv + 0.8, b.get_y() + b.get_height() / 2, f"{vv:.1f}%", va="center", fontsize=7.0, color=INK2)
        ax.text(0.6, b.get_y() + b.get_height() + 0.16, k, va="bottom", fontsize=7.0,
                color=INK if "ours" in k else INK2, fontweight="bold" if "ours" in k else "normal")
    ax.set_yticks([]); ax.set_ylim(-0.6, len(ks) - 0.15)
    ax.set_xlim(0, 44); ax.set_xticks([0, 20, 40])
    ax.set_xlabel("Net $R^2$ lost to the distractor\n(lower is better)", fontsize=7.0, labelpad=2)
    ax.set_title("measured on OXE validation", fontsize=7.0, color=INK2, pad=4, loc="left", style="italic")
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=6.8, length=2.5, pad=1.8, colors=INK2)
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png"); plt.close(fig)


if __name__ == "__main__":
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "figures")
    nuisance(os.path.join(d, "nuisance_examples"))
    print("wrote", d + "/nuisance_examples.{pdf,png}")
