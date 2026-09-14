"""Shared plot styling.  Palette taken from the validated categorical set
(slots 1-5, light surface); it clears the lightness, chroma, CVD-separation and
normal-vision gates on the adjacent pairlist.

Three slots sit below 3:1 contrast on the light surface, so the *relief rule*
applies: every chart here ships visible value labels and/or distinct markers,
and the full numbers are always written to the matching results/*.csv (the
table view).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "figure.dpi": 160,
    })


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, zorder=0)
    ax.set_axisbelow(True)


def bar_labels(ax, bars, fmt="{:.2f}", dy=0.012):
    """Visible value labels -- the relief for sub-3:1 series colors."""
    for b in bars:
        h = b.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2,
                                    h + (dy if h >= 0 else -dy)),
                    ha="center", va=va, fontsize=7, color=INK2)


def save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path}")
