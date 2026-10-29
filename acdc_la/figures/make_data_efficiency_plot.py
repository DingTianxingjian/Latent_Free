"""LIBERO success rate vs fraction of training demonstrations, in the layout of UniVLA's Fig. 10 (2026-10-27).

UniVLA and OpenVLA points: UniVLA README / paper Fig. 10 (Goal, Long) and Table I (full data).  Ours: Newton runs with
the 1/10-scale LoRA phase 2 (latp2_inv labels), 500 episodes each (acdc_la/results/vla_eval/merged_*nw_p2ours_*).
The hollow Goal point is the 2026-09-19 workstation run WITHOUT phase 2 (ledger), shown separately because its
setup differs.  Usage:  python acdc_la/figures/make_data_efficiency_plot.py  -> acdc_la/figures/libero_data_efficiency.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libero_data_efficiency.png")
# PROJECTED (not measured) 20 % / 50 % points for Ours: UniVLA's own curve shape mapped onto our two measured endpoints,
#   ours(f) = ours(10) + (univla(f) - univla(10)) / (univla(100) - univla(10)) * (ours(100) - ours(10))
# Drawn hollow on a dashed line and labelled "projected"; replace with measured runs (20 % / 50 %) when they exist.
def project(univla, lo, hi):
    (x, y), u = univla, dict(zip(*univla))
    return [(f, lo + (u[f] - u[10]) / (u[100] - u[10]) * (hi - lo)) for f in (20, 50)]
# validated default categorical palette, slots 1-3 in fixed order (dataviz skill, references/palette.md)
C_UNIVLA, C_OURS, C_OPENVLA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

panels = {
    "LIBERO-Goal": dict(
        univla=([10, 20, 50, 100], [86.3, 90.4, 93.1, 95.6]),
        openvla=([10, 20, 50, 100], [61.4, 66.0, 77.0, 79.2]),
        ours=([100], [94.4]),
        ours_nop2=(10, 86.8),
        proj=(86.8, 94.4)),
    "LIBERO-Long": dict(
        univla=([10, 20, 50, 100], [62.4, 71.4, 87.0, 92.0]),
        openvla=([10, 20, 50, 100], [11.6, 22.4, 36.6, 53.7]),
        ours=([10, 100], [61.6, 89.0]),
        proj=(61.6, 89.0)),
    "LIBERO-Object": dict(
        univla=([100], [96.8]),
        openvla=([100], [88.4]),
        ours=([10, 100], [88.4, 98.8])),
}

plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "font.family": "DejaVu Sans"})
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURF)
for ax, (name, d) in zip(axes, panels.items()):
    ax.set_facecolor(SURF)
    for key, color, label in (("openvla", C_OPENVLA, "OpenVLA (UniVLA paper)"), ("univla", C_UNIVLA, "UniVLA (paper)"),
                              ("ours", C_OURS, "Ours (measured)")):
        x, y = d[key]
        if key == "ours":
            pts = sorted(list(zip(x, y)) + (project(d["univla"], *d["proj"]) if "proj" in d else [])
                         + ([d["ours_nop2"]] if "ours_nop2" in d else []))
            ax.plot(*zip(*pts), color=color, lw=2, ls=(0, (4, 3)), zorder=3)
            if "proj" in d:
                px, py = zip(*project(d["univla"], *d["proj"]))
                ax.scatter(px, py, s=64, facecolor=SURF, edgecolor=color, linewidth=2, marker="D", zorder=4,
                           label="Ours (projected)")
                for xi, yi in zip(px, py):
                    ax.annotate(f"~{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, -17), ha="center",
                                fontsize=9.5, color=INK2)
        else:
            ax.plot(x, y, color=color, lw=2, solid_capstyle="round", zorder=3)
        ax.scatter(x, y, s=64, color=color, edgecolor=SURF, linewidth=2, zorder=4, label=label)
        for xi, yi in zip(x, y):
            if key != "openvla":                       # label the two curves the story is about, not every point
                other = dict(zip(*d["univla" if key == "ours" else "ours"]))
                above = xi not in other or yi >= other[xi]          # the higher of two close points labels above, the lower below
                ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, 9 if above else -17),
                            ha="center", fontsize=9.5, color=INK)
    if "ours_nop2" in d:
        x, y = d["ours_nop2"]
        ax.scatter([x], [y], s=64, facecolor=SURF, edgecolor=C_OURS, linewidth=2, zorder=4,
                   label="Ours, no phase 2 (Sep 19)")
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(-10, -3), ha="right", fontsize=9.5, color=INK2)
    ax.set_title(name, fontsize=13, color=INK, fontweight="bold", loc="left")
    ax.set_xticks([10, 20, 50, 100]); ax.set_xlim(-2, 107)
    ax.set_xlabel("Training demos (%)")
    ax.grid(True, color=GRID, lw=1, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("Success rate (%)")
axes[0].set_ylim(55, 100); axes[1].set_ylim(0, 100); axes[2].set_ylim(80, 101)
h, l = axes[0].get_legend_handles_labels()
h, l = [h[i] for i in (0, 1, 3, 2, 4)], [l[i] for i in (0, 1, 3, 2, 4)]
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("LIBERO success rate vs. fraction of training demonstrations (500 episodes per point)",
             x=0.01, ha="left", fontsize=13, color=INK)
fig.text(0.01, 0.905, "Ours: phase 2 at 1/10 scale (LoRA).  Filled = measured (500 episodes); hollow diamonds = projected from "
         "UniVLA's curve shape between our measured endpoints (runs in progress).  Goal 10 %: measured without phase 2.",
         fontsize=9.5, color=INK2)
fig.tight_layout(rect=(0, 0.07, 1, 0.9))
fig.savefig(OUT, dpi=160, facecolor=SURF)
print("wrote", OUT)
