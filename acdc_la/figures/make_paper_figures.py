"""Paper figures: figures/representation_summary.pdf and figures/eat_refinement.pdf.

    envs/univla_vla/bin/python -m acdc_la.figures.make_paper_figures        (or run the file directly)

EVERY number lives in the DATA block below, tagged with its provenance, so swapping in a final value is a one-line edit:
  [real]        measured, with the LEDGER date it comes from
  [PLACEHOLDER] not measured yet -- replace before submission
The probe numbers are R^2 on the OXE validation split (per-source episode-disjoint ridge, 400 clips/source, 29 sources);
`drop` is the relative loss of Net R^2 under the moving-patch distractor, computed as the ratio of means.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --------------------------------------------------------------------------------------------------- DATA
# Frozen tokenizers on the OXE validation split.                      [real] LEDGER 2026-09-11 / 09-14 / 09-16
TOKENIZERS = {                       # label:            (Net,   Seg1,  Profile, Seq,   drop %)
    "Pooled temporal":               (0.336, 0.210, 0.017, 0.200, 33.6),
    "ACDC (earlier tokenizer)":      (0.324, 0.264, 0.085, 0.224, 28.9),
    "UniVLA (matched, 1 ep)":        (0.236, 0.222, 0.125, 0.215, 22.9),
    "UniVLA (matched, 2 ep)":        (0.261, 0.249, 0.139, 0.237, 18.2),
    "TERRA (ours)":                  (0.329, 0.283, 0.091, 0.236, 17.9),
}
# Panel (a): the paper cites the NMSE action probe of Lee et al.  [PLACEHOLDER protocol] we have not run theirs; shown
# here as 1 - R^2 of OUR ridge probe on the full 12-step action sequence (the same quantity for a standardised target,
# a different probe).  Replace this dict with the measured NMSE; nothing else in the file changes.
NMSE = {k: round(1.0 - v[3], 3) for k, v in TOKENIZERS.items()}

# Controlled EAT continuation: all arms start from the same 65k-step checkpoint and get the same 15k extra steps.
EAT_START = (0.265, 0.232, 0.091, 0.192, 22.4)          # [real] LEDGER 2026-09-16 (`tc_cont`, the shared start)
EAT = {                              # label:                        (Net,   Seg1,  Profile, Seq,   drop %)
    "Reconstruction only":           (0.268, 0.240, 0.091, 0.193, 22.6),   # [real] `tc_cont_plus15k`
    "EAT w/o nuisance invariance":   (0.300, 0.267, 0.085, 0.216, 19.7),   # [real] `latp2_anchor`
    "EAT (complete)":                (0.329, 0.283, 0.091, 0.236, 17.9),   # [real] `latp2_inv`
}
TARGETS = ["Net", "Seg1", "Profile", "Seq"]
SHORT = {"Pooled temporal": "Pooled", "ACDC (earlier tokenizer)": "ACDC", "UniVLA (matched, 1 ep)": "UniVLA 1 ep",
         "UniVLA (matched, 2 ep)": "UniVLA 2 ep", "TERRA (ours)": "TERRA (ours)"}

# --------------------------------------------------------------------------------------------------- STYLE
C = {"pooled": "#1baf7a", "acdc": "#eb6834", "univla": "#2a78d6", "ours": "#4a3aa7"}   # validated (dataviz skill)
RAMP = ["#b0a5e2", "#7563c0", "#4a3aa7"]                                  # sequential: refinement progression
INK, INK2, GRID = "#1a1a19", "#52514e", "#d9d9d6"
SERIES_COLOR = [C["pooled"], C["acdc"], C["univla"], C["univla"], C["ours"]]
SERIES_HATCH = ["", "", "", "///", ""]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7.2,
    "xtick.labelsize": 7.4, "ytick.labelsize": 7.4, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.edgecolor": INK2, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "figure.dpi": 160, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
})


def tidy(ax, xgrid=False, ygrid=False):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    if xgrid: ax.xaxis.grid(True, color=GRID, lw=0.5)
    if ygrid: ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.tick_params(length=2.5, pad=1.8)


def panel_tag(ax, tag, title):
    ax.set_title(f"({tag}) {title}", loc="left", pad=6, color=INK, fontweight="bold")


def hbars(ax, labels, values, colors, hatches, fmt, xlabel, better, xmax=None):
    """Horizontal bars, best at the top, value at each bar end (the relief the palette's low-contrast slot needs)."""
    y = range(len(labels))
    bars = ax.barh(list(y), values, height=0.62, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    for b, h in zip(bars, hatches):
        if h: b.set_hatch(h); b.set_edgecolor("white")
    ax.set_yticks(list(y)); ax.set_yticklabels(labels)
    for lab, t in zip(ax.get_yticklabels(), labels):
        if "ours" in t: lab.set_fontweight("bold"); lab.set_color(INK)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax or max(values) * 1.22)
    for b, v in zip(bars, values):
        ax.text(b.get_width() + ax.get_xlim()[1] * 0.018, b.get_y() + b.get_height() / 2, fmt.format(v),
                va="center", ha="left", fontsize=7.2, color=INK2)
    ax.set_xlabel(f"{xlabel}\n({better})", labelpad=2)
    tidy(ax, xgrid=True)


def dots(ax, labels, values, colors, fmt, xlabel, better, pad=0.012):
    """Dot plot: the values sit in a narrow band, where a bar's mandatory zero baseline would hide the differences."""
    y = list(range(len(labels)))
    lo, hi = min(values), max(values)
    ax.hlines(y, lo - pad, values, color=GRID, lw=1.0, zorder=2)
    for yi, v, c in zip(y, values, colors):
        ax.plot([v], [yi], "o", ms=6.5, color=c, markeredgecolor="white", markeredgewidth=0.9, zorder=3)
        ax.text(v, yi - 0.34, fmt.format(v), ha="center", va="bottom", fontsize=7.2, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    for lab, t in zip(ax.get_yticklabels(), labels):
        if "ours" in t: lab.set_fontweight("bold"); lab.set_color(INK)
    ax.invert_yaxis(); ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(len(labels) - 0.4, -0.75)
    ax.set_xlabel(f"{xlabel}\n({better})", labelpad=2)
    tidy(ax, xgrid=True)


def grouped(ax, groups, series, ylabel, ncol=3, legend=True, ylim=None):
    """series: [(label, [v per group], color, hatch)] -- grouped bars with a 2 px surface gap between neighbours."""
    n, g = len(series), len(groups)
    w = 0.78 / n
    for i, (lab, vals, col, hat) in enumerate(series):
        xs = [j - 0.39 + w * (i + 0.5) for j in range(g)]
        bars = ax.bar(xs, vals, width=w * 0.88, color=col, label=lab, edgecolor="white", linewidth=0.7, zorder=3)
        if hat:
            for b in bars: b.set_hatch(hat); b.set_edgecolor("white")
    ax.set_xticks(range(g)); ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    tidy(ax, ygrid=True)
    if legend:
        ax.legend(ncol=ncol, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.30),
                  handlelength=1.1, handleheight=0.9, columnspacing=1.2, borderpad=0.2, labelspacing=0.3)


def representation_summary(out):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.0, 1.45, 1.0], "wspace": 0.78})
    labs = list(TOKENIZERS); short = [SHORT[l] for l in labs]

    panel_tag(axes[0], "a", "Action alignment")
    dots(axes[0], short, [NMSE[l] for l in labs], SERIES_COLOR, "{:.3f}", "action probe NMSE", "lower is better")

    panel_tag(axes[1], "b", "Linear temporal readouts")
    grouped(axes[1], TARGETS,
            [(l, list(TOKENIZERS[l][:4]), SERIES_COLOR[i], SERIES_HATCH[i]) for i, l in enumerate(labs)],
            r"probe $R^2$", ncol=3, ylim=(0, 0.40), legend=False)
    axes[1].set_xlabel("(higher is better)", labelpad=2)

    panel_tag(axes[2], "c", "Nuisance robustness")
    hbars(axes[2], short, [TOKENIZERS[l][4] for l in labs], SERIES_COLOR, SERIES_HATCH, "{:.1f}%",
          r"Net $R^2$ lost to the distractor", "lower is better", xmax=44)

    handles = [Patch(facecolor=SERIES_COLOR[i], edgecolor="white", hatch=SERIES_HATCH[i], label=l)
               for i, l in enumerate(labs)]
    fig.legend(handles=handles, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10),
               handlelength=1.1, handleheight=0.9, columnspacing=1.4, borderpad=0.2)
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png"); plt.close(fig)


def eat_refinement(out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.42})
    labs = list(EAT)

    panel_tag(axes[0], "a", "Temporal readouts after refinement")
    grouped(axes[0], TARGETS, [(l, list(EAT[l][:4]), RAMP[i], "") for i, l in enumerate(labs)],
            r"probe $R^2$  (higher is better)", ncol=3, ylim=(0, 0.40))
    for j, v in enumerate(EAT[labs[-1]][:4]):                  # direct labels on the complete method only
        axes[0].text(j + 0.26, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.9, color=INK2)
    for j, v in enumerate(EAT_START[:4]):                       # the shared starting checkpoint, as a reference tick
        axes[0].plot([j - 0.42, j + 0.42], [v, v], color=INK2, lw=1.0, ls=(0, (3, 2)), zorder=4,
                     label="Shared start checkpoint" if j == 0 else None)
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(h, l, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.30),
                   handlelength=1.3, columnspacing=1.0, borderpad=0.2)

    panel_tag(axes[1], "b", "Nuisance robustness")
    vals = [EAT[l][4] for l in labs]
    bars = axes[1].bar(range(len(labs)), vals, width=0.58, color=RAMP, edgecolor="white", linewidth=0.8, zorder=3)
    axes[1].axhline(EAT_START[4], color=INK2, lw=1.0, ls=(0, (3, 2)), zorder=4, xmax=0.88)
    axes[1].text(1.9, EAT_START[4] + 0.6, "shared start", fontsize=6.8, color=INK2, ha="right")
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=7.2, color=INK2)
    xa = 2.62                                                  # the delta arrow sits clear of every bar and label
    axes[1].annotate("", xy=(xa, vals[2]), xytext=(xa, EAT_START[4]),
                     arrowprops=dict(arrowstyle="-|>", color=C["ours"], lw=1.1, shrinkA=0, shrinkB=0))
    axes[1].text(xa + 0.12, (vals[2] + EAT_START[4]) / 2, f"$-${EAT_START[4] - vals[2]:.1f}\npp", fontsize=7.2,
                 color=C["ours"], ha="left", va="center", fontweight="bold", linespacing=0.95)
    axes[1].set_xticks(range(len(labs)))
    axes[1].set_xticklabels(["Recon.\nonly", "EAT\nw/o inv.", "EAT\nfull"], fontsize=7.2)
    axes[1].set_ylabel(r"Net $R^2$ lost to the distractor (%)")
    axes[1].set_xlabel("(lower is better)", labelpad=2)
    axes[1].set_ylim(0, 27); axes[1].set_xlim(-0.55, 3.05)
    tidy(axes[1], ygrid=True)
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png"); plt.close(fig)


if __name__ == "__main__":
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "figures")
    os.makedirs(d, exist_ok=True)
    representation_summary(os.path.join(d, "representation_summary"))
    eat_refinement(os.path.join(d, "eat_refinement"))
    print("wrote", d + "/{representation_summary,eat_refinement}.{pdf,png}")
