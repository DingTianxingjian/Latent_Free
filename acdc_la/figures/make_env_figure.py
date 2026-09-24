"""Environment figure: figures/libero_suites.{pdf,png} and figures/libero_suites_grid.{pdf,png}.

    ../envs/univla_vla/bin/python -m acdc_la.figures.make_env_figure

Reads the frames rendered by acdc_la/figures/grab_libero_frames.py (run on a GPU node through
acdc_la/scripts/run_libero_frames.sh) plus their instructions, and lays them out two ways:
  libero_suites       one frame per suite, a compact full-width strip
  libero_suites_grid  two tasks per suite, when there is room for it
Both are real renders of the initial state the policy sees, not illustrations.
"""
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

INK, INK2 = "#1a1a19", "#52514e"
TITLES = {"libero_spatial": "LIBERO-Spatial", "libero_object": "LIBERO-Object",
          "libero_goal": "LIBERO-Goal", "libero_10": "LIBERO-Long"}
ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 7.6,
    "figure.dpi": 160, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "text.color": INK,
})


def _cap(ax, text, width=30, lines=2):
    w = textwrap.wrap(text.strip().rstrip("."), width=width)
    if len(w) > lines:
        w = w[:lines]; w[-1] = w[-1] + "…"
    ax.set_xlabel("\n".join(w), fontsize=6.6, color=INK2, labelpad=3, linespacing=1.25)


def _show(ax, d, rec):
    ax.imshow(Image.open(os.path.join(d, rec["file"])))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#c9c9c6"); s.set_linewidth(0.6)


def strip(frames, d, out, row=0):
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.35), gridspec_kw={"wspace": 0.10})
    for ax, s in zip(axes, ORDER):
        rec = [r for r in frames if r["suite"] == s][row]
        _show(ax, d, rec)
        ax.set_title(TITLES[s], fontsize=8.2, fontweight="bold", pad=4, color=INK)
        _cap(ax, rec["instruction"])
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png"); plt.close(fig)


def grid(frames, d, out, rows=2):
    fig, axes = plt.subplots(rows, 4, figsize=(7.2, 1.95 * rows + 0.35),
                             gridspec_kw={"wspace": 0.10, "hspace": 0.30})
    for j, s in enumerate(ORDER):
        recs = [r for r in frames if r["suite"] == s]
        for i in range(rows):
            ax = axes[i][j]
            _show(ax, d, recs[i])
            if i == 0:
                ax.set_title(TITLES[s], fontsize=8.2, fontweight="bold", pad=4, color=INK)
            _cap(ax, recs[i]["instruction"], width=26)
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png"); plt.close(fig)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    d = os.path.join(root, "figures", "libero_frames")
    frames = json.load(open(os.path.join(d, "frames.json")))
    strip(frames, d, os.path.join(root, "figures", "libero_suites"))
    if min(len([r for r in frames if r["suite"] == s]) for s in ORDER) >= 2:
        grid(frames, d, os.path.join(root, "figures", "libero_suites_grid"))
    print("wrote", os.path.join(root, "figures") + "/libero_suites{,_grid}.{pdf,png}")
