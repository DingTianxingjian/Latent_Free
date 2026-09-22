"""Paper figures.  Three files in figures/:

    representation_comparison.{pdf,png}   external comparison  -- TERRA vs other latent-action methods
    representation_ablation.{pdf,png}     our own ablation     -- no external method appears here
    eat_refinement.{pdf,png}              objective ablation   -- the EAT terms, on one shared checkpoint

    envs/univla_vla/bin/python -m acdc_la.figures.make_paper_figures        (or run the file directly)

EVERY number lives in the DATA block below, tagged with its provenance, so swapping in a final value is a one-line edit:
  [real]        measured, with the LEDGER date it comes from
  [PLACEHOLDER] invented for the mock-up -- replace before submission; drawn HATCHED so no reader mistakes it
The probe numbers are R^2 on the OXE validation split (per-source episode-disjoint ridge, 400 clips/source, 29 sources);
`drop` is the relative loss of Net R^2 under the moving-patch distractor, computed as the ratio of means.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ------------------------------------------------------------------------- DATA 1: external comparison (Fig. 1)
# Frozen latent-action tokenizers, probed identically.  OpenVLA is NOT here: it regresses/discretises the true action
# directly and has no latent action code to probe -- it appears only in the closed-loop panel (d).
COMPARE = {                      # label:        (Net,   Seg1,  Profile, Seq,   drop %)  provenance
    "LAPA":                      (0.212, 0.191, 0.102, 0.181, 27.5),   # [PLACEHOLDER] not run
    "UniVLA (matched budget)":   (0.236, 0.222, 0.125, 0.215, 22.9),   # [real] LEDGER 2026-09-14, s2mix_1ep
    "TERRA (ours)":              (0.329, 0.283, 0.091, 0.236, 17.9),   # [real] LEDGER 2026-09-16, latp2_inv
}
CMP_FAKE = {"LAPA": True, "UniVLA (matched budget)": False, "TERRA (ours)": False}
CMP_SHORT = {"LAPA": "LAPA", "UniVLA (matched budget)": "UniVLA", "TERRA (ours)": "TERRA (ours)"}
# Panel (a): [PLACEHOLDER protocol] shown as 1 - R^2 of our ridge probe on the full 12-step action sequence.
CMP_NMSE = {k: round(1.0 - v[3], 3) for k, v in COMPARE.items()}

SUITES = ["Spatial", "Object", "Goal", "Long"]
# Closed-loop LIBERO success at 10 % demonstrations, identical VLA recipe.  [PLACEHOLDER] every value in this panel is
# the *expected* shape, not a measurement -- the whole panel is hatched.  Replace suite by suite as the runs land.
CLOSED = {
    "OpenVLA (no latent)":       (71.0, 69.5, 67.0, 41.0),
    "LAPA":                      (76.0, 74.0, 74.5, 45.0),
    "UniVLA (matched budget)":   (84.2, 82.8, 86.6, 48.0),
    "TERRA (ours)":              (87.5, 85.0, 88.0, 52.0),
}

# ------------------------------------------------------------------------- DATA 2: our ablation ladder (Fig. 2)
# Same tokenizer family, same data and budget; only the temporal/spatial factorisation changes.  No external method.
SHOW_CELLWISE = False            # the earlier cell-wise tokenizer is cut from the paper; True restores the middle rung
ABLATE = {                       # label:        (Net,   Seg1,  Profile, Seq,   drop %)
    "Mean-pooled window":        (0.336, 0.210, 0.017, 0.200, 33.6),   # [real] s1_mean_1ep: no temporal decomposition
    "Cell-wise (earlier)":       (0.324, 0.264, 0.085, 0.224, 28.9),   # [real] s1_acdc_1ep: per-cell MLP  -- cut
    "TERRA (ours)":              (0.329, 0.283, 0.091, 0.236, 17.9),   # [real] latp2_inv
}
if not SHOW_CELLWISE:
    ABLATE.pop("Cell-wise (earlier)")
ABL_NMSE = {k: round(1.0 - v[3], 3) for k, v in ABLATE.items()}

# ------------------------------------------------------------------------- DATA 3: EAT objective ablation (Fig. 3)
EAT_START = (0.265, 0.232, 0.091, 0.192, 22.4)          # [real] LEDGER 2026-09-16 (`tc_cont`, the shared start)
EAT = {                              # label:                        (Net,   Seg1,  Profile, Seq,   drop %)
    "Reconstruction only":           (0.268, 0.240, 0.091, 0.193, 22.6),   # [real] `tc_cont_plus15k`
    "EAT w/o nuisance invariance":   (0.300, 0.267, 0.085, 0.216, 19.7),   # [real] `latp2_anchor`
    "EAT (complete)":                (0.329, 0.283, 0.091, 0.236, 17.9),   # [real] `latp2_inv`
}
TARGETS = ["Net", "Seg1", "Profile", "Seq"]

# ------------------------------------------------------------------------- STYLE
C = {"openvla": "#1baf7a", "lapa": "#eb6834", "univla": "#2a78d6", "ours": "#4a3aa7"}   # validated (dataviz skill)
RAMP = ["#b0a5e2", "#7563c0", "#4a3aa7"]                    # sequential: one hue, light -> dark (our own ladder)
INK, INK2, GRID = "#1a1a19", "#52514e", "#d9d9d6"
CMP_COLOR = [C["lapa"], C["univla"], C["ours"]]
CLOSED_COLOR = [C["openvla"], C["lapa"], C["univla"], C["ours"]]
FAKE_HATCH = "///"

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


def bold_ours(ax):
    for lab in ax.get_yticklabels():
        if "ours" in lab.get_text(): lab.set_fontweight("bold"); lab.set_color(INK)


def hbars(ax, labels, values, colors, hatches, fmt, xlabel, better, xmax=None):
    """Horizontal bars, value at each bar end (the relief the palette's low-contrast slot needs)."""
    y = list(range(len(labels)))
    bars = ax.barh(y, values, height=0.6, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    for b, h in zip(bars, hatches):
        if h: b.set_hatch(h); b.set_edgecolor("white")
    ax.set_yticks(y); ax.set_yticklabels(labels); bold_ours(ax)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax or max(values) * 1.22)
    for b, v in zip(bars, values):
        ax.text(b.get_width() + ax.get_xlim()[1] * 0.018, b.get_y() + b.get_height() / 2, fmt.format(v),
                va="center", ha="left", fontsize=7.2, color=INK2)
    ax.set_xlabel(f"{xlabel}\n({better})", labelpad=2)
    tidy(ax, xgrid=True)


def dots(ax, labels, values, colors, hatches, fmt, xlabel, better, pad=0.012):
    """Dot plot: the values sit in a narrow band, where a bar's mandatory zero baseline would hide the differences."""
    y = list(range(len(labels)))
    lo, hi = min(values), max(values)
    ax.hlines(y, lo - pad, values, color=GRID, lw=1.0, zorder=2)
    for yi, v, c, h in zip(y, values, colors, hatches):
        ax.plot([v], [yi], "o", ms=6.8, color="white" if h else c, markeredgecolor=c,
                markeredgewidth=1.6 if h else 0.9, zorder=3)
        ax.text(v, yi - 0.34, fmt.format(v), ha="center", va="bottom", fontsize=7.2, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(labels); bold_ours(ax)
    ax.invert_yaxis(); ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(len(labels) - 0.45, -0.8)
    ax.set_xlabel(f"{xlabel}\n({better})", labelpad=2)
    tidy(ax, xgrid=True)


def grouped(ax, groups, series, ylabel, ncol=3, legend=True, ylim=None, label_last=False, fmt="{:.3f}"):
    """series: [(label, [v per group], color, hatch)] -- grouped bars with a 2 px surface gap between neighbours."""
    n, g = len(series), len(groups)
    w = 0.78 / n
    for i, (lab, vals, col, hat) in enumerate(series):
        xs = [j - 0.39 + w * (i + 0.5) for j in range(g)]
        bars = ax.bar(xs, vals, width=w * 0.88, color=col, label=lab, edgecolor="white", linewidth=0.7, zorder=3)
        if hat:
            for b in bars: b.set_hatch(hat); b.set_edgecolor("white")
        if label_last and i == n - 1:                      # label our bar, but clear of a taller neighbour
            top = [max(sv[1][j] for sv in series) for j in range(g)]
            for j, (b, v) in enumerate(zip(bars, vals)):
                ax.text(b.get_x() + b.get_width() / 2, top[j] + (ylim[1] - ylim[0]) * 0.025, fmt.format(v),
                        ha="center", fontsize=6.9, color=INK2)
    ax.set_xticks(range(g)); ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    tidy(ax, ygrid=True)
    if legend:
        ax.legend(ncol=ncol, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.30),
                  handlelength=1.1, handleheight=0.9, columnspacing=1.2, borderpad=0.2, labelspacing=0.3)


def representation_comparison(out, closed_loop=True):
    """Comparison with other latent-action methods.

    closed_loop=False -> panels (a-c) only: the main-text figure, where Table 1 already carries the success rates.
    closed_loop=True  -> adds panel (d), the policy-level view; for the teaser slot or a talk, not next to Table 1.
    """
    if closed_loop:
        fig = plt.figure(figsize=(7.2, 4.4))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.5], height_ratios=[1.0, 1.0], wspace=0.60, hspace=0.72)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    else:
        fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.0, 1.45, 1.0], "wspace": 0.72})
        axes = list(axs)
    labs = list(COMPARE); short = [CMP_SHORT[l] for l in labs]
    hatch = [FAKE_HATCH if CMP_FAKE[l] else "" for l in labs]

    panel_tag(axes[0], "a", "Action alignment")
    dots(axes[0], short, [CMP_NMSE[l] for l in labs], CMP_COLOR, hatch, "{:.3f}", "action probe NMSE", "lower is better")

    panel_tag(axes[1], "b", "Linear temporal readouts")
    grouped(axes[1], TARGETS, [(CMP_SHORT[l], list(COMPARE[l][:4]), CMP_COLOR[i], hatch[i]) for i, l in enumerate(labs)],
            r"probe $R^2$", ylim=(0, 0.40), legend=False, label_last=True)
    axes[1].set_xlabel("(higher is better)", labelpad=2)

    panel_tag(axes[2], "c", "Nuisance robustness")
    hbars(axes[2], short, [COMPARE[l][4] for l in labs], CMP_COLOR, hatch, "{:.1f}%",
          r"Net $R^2$ lost to the distractor", "lower is better", xmax=40)

    clabs = list(CLOSED)
    if not closed_loop:                       # main text: legend of the three probed tokenizers, no policy panel
        handles = [Patch(facecolor=CMP_COLOR[i], edgecolor="white", hatch=FAKE_HATCH if CMP_FAKE[l] else "",
                         label=CMP_SHORT[l]) for i, l in enumerate(labs)]
        handles += [Patch(facecolor="white", edgecolor=INK2, hatch=FAKE_HATCH, linewidth=0.6,
                          label="expected value (not yet measured)")]
        lg = fig.legend(handles=handles, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10),
                        handlelength=1.2, handleheight=0.9, columnspacing=1.4, borderpad=0.2)
        for t in lg.get_texts():
            if "expected" in t.get_text(): t.set_color(INK2); t.set_style("italic")
        fig.savefig(out + ".pdf"); fig.savefig(out + ".png"); plt.close(fig)
        return

    panel_tag(axes[3], "d", "Closed-loop LIBERO, 10 % demonstrations")
    grouped(axes[3], SUITES, [(l, list(CLOSED[l]), CLOSED_COLOR[i], FAKE_HATCH) for i, l in enumerate(clabs)],
            "success rate (%)", ylim=(0, 100), legend=False)
    for i, l in enumerate(clabs):                      # every bar labelled: the relief the green slot's contrast needs
        w = 0.78 / len(clabs)
        for j, v in enumerate(CLOSED[l]):
            axes[3].text(j - 0.39 + w * (i + 0.5), v + 1.5, f"{v:.0f}", ha="center", fontsize=6.2, color=INK2,
                         rotation=90, va="bottom")
    axes[3].set_xlabel("(higher is better)", labelpad=2)

    handles = [Patch(facecolor=CLOSED_COLOR[i], edgecolor="white", label=l.replace(" (matched budget)", ""))
               for i, l in enumerate(clabs)]
    handles += [Patch(facecolor="white", edgecolor=INK2, hatch=FAKE_HATCH, linewidth=0.6,
                      label="expected value (not yet measured)")]
    lg = fig.legend(handles=handles, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.05),
                    handlelength=1.2, handleheight=0.9, columnspacing=1.4, borderpad=0.2)
    for t in lg.get_texts():
        if "expected" in t.get_text(): t.set_color(INK2); t.set_style("italic")
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png"); plt.close(fig)


def representation_ablation(out):
    """Fig. 2 -- our tokenizer only: what the temporal factorisation buys.  No external method appears."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1.0, 1.45, 1.0], "wspace": 0.72})
    labs = list(ABLATE)
    short = {"Mean-pooled window": "Mean-pooled", "Cell-wise (earlier)": "Cell-wise\n(earlier)",
             "TERRA (ours)": "TERRA (ours)"}
    short = [short[l] for l in labs]
    ramp = RAMP if len(labs) == 3 else [RAMP[0], RAMP[2]]
    nohatch = [""] * len(labs)

    panel_tag(axes[0], "a", "Action alignment")
    dots(axes[0], short, [ABL_NMSE[l] for l in labs], ramp, nohatch, "{:.3f}", "action probe NMSE", "lower is better")

    panel_tag(axes[1], "b", "Linear temporal readouts")
    grouped(axes[1], TARGETS, [(l, list(ABLATE[l][:4]), RAMP[i], "") for i, l in enumerate(labs)],
            r"probe $R^2$", ylim=(0, 0.40), legend=False, label_last=True)
    axes[1].set_xlabel("(higher is better)", labelpad=2)

    panel_tag(axes[2], "c", "Nuisance robustness")
    hbars(axes[2], short, [ABLATE[l][4] for l in labs], ramp, nohatch, "{:.1f}%",
          r"Net $R^2$ lost to the distractor", "lower is better", xmax=42)

    handles = [Patch(facecolor=ramp[i], edgecolor="white", label=l) for i, l in enumerate(labs)]
    fig.legend(handles=handles, ncol=len(labs), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10),
               handlelength=1.1, handleheight=0.9, columnspacing=1.6, borderpad=0.2)
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png"); plt.close(fig)


def eat_refinement(out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.42})
    labs = list(EAT)

    panel_tag(axes[0], "a", "Temporal readouts after refinement")
    grouped(axes[0], TARGETS, [(l, list(EAT[l][:4]), RAMP[i], "") for i, l in enumerate(labs)],
            r"probe $R^2$  (higher is better)", ncol=3, ylim=(0, 0.40), label_last=True)
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
    representation_comparison(os.path.join(d, "representation_comparison"), closed_loop=False)
    representation_comparison(os.path.join(d, "representation_comparison_with_policy"), closed_loop=True)
    representation_ablation(os.path.join(d, "representation_ablation"))
    eat_refinement(os.path.join(d, "eat_refinement"))
    print("wrote", d + "/{representation_comparison,representation_comparison_with_policy,"
          "representation_ablation,eat_refinement}.{pdf,png}")
