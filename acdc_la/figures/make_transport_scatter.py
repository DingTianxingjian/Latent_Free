"""Per-source transported-action NMSE: reconstruction-only (x) vs full EAT (y), one dot per OXE source.

    ../envs/univla_vla/bin/python -m acdc_la.figures.make_transport_scatter

Reads acdc_la/results/transport_by_source.json (acdc_la/downstream/transport_by_source.py).  Far donors: the mean of
each source's Q3 and Q4 donor-distance quartiles.  Below the diagonal = EAT transports that source's effects better."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, INK2, GRID = "#1a1a19", "#52514e", "#d9d9d6"
OURS, OTHER = "#4a3aa7", "#b5b5b1"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 7.6, "figure.dpi": 160, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "text.color": INK,
})


def far(row):
    v = [row[k] for k in ("Q3", "Q4") if k in row]
    return float(np.mean(v)) if v else np.nan


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    d = json.load(open(os.path.join(root, "acdc_la", "results", "transport_by_source.json")))
    S = sorted(set(d["recon"]) & set(d["eat"]))
    x = np.array([far(d["recon"][s]) for s in S]); y = np.array([far(d["eat"][s]) for s in S])
    n = np.array([d["eat"][s]["n_test"] for s in S], float)
    ok = np.isfinite(x) & np.isfinite(y); x, y, n = x[ok], y[ok], n[ok]
    below = y < x
    fig, ax = plt.subplots(figsize=(3.3, 3.1))
    lo, hi = min(x.min(), y.min()) - 0.03, max(x.max(), y.max()) + 0.03
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], color="#efedf8", zorder=0)          # EAT-better half
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.9, ls=(0, (4, 3)), zorder=1)
    size = 14 + 26 * (n - n.min()) / max(n.max() - n.min(), 1)
    ax.scatter(x[~below], y[~below], s=size[~below], color=OTHER, edgecolors="white", linewidths=0.7, zorder=3)
    ax.scatter(x[below], y[below], s=size[below], color=OURS, edgecolors="white", linewidths=0.7, zorder=4)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("reconstruction only: transported-action NMSE", labelpad=3)
    ax.set_ylabel("EAT: transported-action NMSE", labelpad=3)
    ax.text(0.97, 0.05, f"EAT lower on {int(below.sum())} / {len(x)} sources", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.6, color=OURS, fontweight="bold")
    ax.text(0.04, 0.95, "far donors (Q3–Q4)", transform=ax.transAxes, ha="left", va="top", fontsize=7.2,
            color=INK2, style="italic")
    for sd in ("top", "right"): ax.spines[sd].set_visible(False)
    for sd in ("left", "bottom"): ax.spines[sd].set_color(INK2); ax.spines[sd].set_linewidth(0.6)
    ax.tick_params(labelsize=6.8, length=2.5, colors=INK2)
    out = os.path.join(root, "figures", "transport_scatter")
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png"); plt.close(fig)
    print(f"far donors: EAT lower on {int(below.sum())}/{len(x)}; median recon {np.median(x):.3f} EAT {np.median(y):.3f}")
    print("wrote figures/transport_scatter.{pdf,png}")


if __name__ == "__main__":
    main()
