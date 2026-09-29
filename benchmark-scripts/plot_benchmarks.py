"""Full-width benchmark figure for the paper: runtime of the original and
AI-optimized implementations in all three case studies.

Reads the results from benchmark-results/ and draws three panels in one row:
openTSNE Barnes-Hut and FIt-SNE (optimization phase, 750 iterations, on 8
threads), ssGSEA (1000 samples, increasing number of gene sets), and graphlet
counting (Erdos-Renyi graphs with 10k nodes, increasing edge count).
Every panel has a linear x and a log y axis. The ssGSEA and graphlet panels
share the t-SNE panel's upper y limit, which leaves headroom above their
highest points. ssGSEA runs longer than 1200 s are not drawn.

The AI-optimized implementation is black in every panel, and the baselines
take saturated pgfplots-style colours, with the naive baseline (naive ssGSEA walk, brute-force
graphlet enumeration) always first. In the t-SNE panel, colour is the
implementation and marker shape the approximation; elsewhere every series has
its own marker, so the figure survives greyscale printing. A missing value (empty CSV cell) is a run that was not performed
because it would have exceeded the time budget.

Typography, line weights, ticks and layout come from paper.mplstyle beside
this script; this script sets only colours, markers, grid and legend
placement.

Usage:
    python plot_benchmarks.py                       # figures/benchmarks.pdf
    python plot_benchmarks.py --out fig.png --width 6.8425 --height 2.5
    python plot_benchmarks.py --tsne-yscale linear --out ../figures/benchmarks-tsne-linear.pdf

The default width, 6.8425 in (494.5 pt), is \\textwidth of the paper's cas-dc layout,
so the figure is placed at 1:1 scale in a figure* environment.

Requires matplotlib.
"""
import argparse
import csv
import math
import os
from os.path import abspath, dirname, join

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
from matplotlib.ticker import LogLocator, NullFormatter, StrMethodFormatter

ROOT = dirname(dirname(abspath(__file__)))
RESULTS = join(ROOT, "benchmark-results")
STYLE = join(dirname(abspath(__file__)), "paper.mplstyle")

# The AI implementation is black; baselines take saturated colours in fixed
# order, after pgfplots' default cycle (blue, red, brown!60!black), with a dark
# green as the fourth. Red and green are hard to tell apart under
# colour-vision deficiency, which is why every series also carries a distinct
# marker.
AI = dict(color="black", marker="o", label="Agent")
NAIVE = dict(color="#0000ff", marker="s")
TOOL_1 = dict(color="#ff0000", marker="^")
TOOL_2 = dict(color="#734d26", marker="D")
TOOL_3 = dict(color="#008000", marker="v")

# Marker per t-SNE approximation.
TSNE_METHODS = (("bh", "^", "Barnes–Hut"), ("fft", "o", "FIt-SNE"))

MUTED = "#52514e"
GRID = "#e4e3df"

# Distance from an axes' bottom edge to the top of its legend: clears the tick
# labels and the x label at the style's 6 pt text. Every panel uses the same
# offset, so the legends share a baseline.
LEGEND_OFFSET_PT = 24


def read_csv(name):
    """Columns of a results file as lists of floats; empty cells become NaN."""
    with open(join(RESULTS, name), newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) if r[k] else math.nan for r in rows] for k in rows[0]}


def series(ax, x, y, style, label=None):
    ax.plot(
        x, y,
        color=style["color"],
        marker=style["marker"],
        markeredgewidth=plt.rcParams["lines.linewidth"],
        label=label if label is not None else style.get("label"),
        clip_on=False,
        zorder=3,
    )


def style_axes(ax, title, xlabel, yscale="log"):
    ax.set_yscale(yscale)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))  # 1,000
    ax.grid(True, which="major", color=GRID, linewidth=0.5, zorder=0)
    if yscale == "log":
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_ylim(bottom=0)


def legend_below(ax, handles, ncol):
    """Legend centred under `ax`, a fixed distance below its bottom edge."""
    offset = ScaledTranslation(0, -LEGEND_OFFSET_PT / 72, ax.figure.dpi_scale_trans)
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        bbox_transform=ax.transAxes + offset,
        ncol=ncol,
        handlelength=2.2,
        columnspacing=1.0,
        labelspacing=0.3,
        borderaxespad=0,
    )


def letter_panels(fig, axes, letters="abc"):
    """Bold panel letters at the left edge of each panel (its tick and axis
    labels included), on one shared baseline: level with the panel titles, or
    higher if a y tick label reaches above them.

    Freezes the layout first: the letters are placed from the final axes
    positions, and must not take part in the layout themselves."""
    fig.canvas.draw()
    fig.set_layout_engine("none")
    renderer = fig.canvas.get_renderer()
    pad = 1.5 / 72 * fig.dpi  # points to display units
    def label_tops(ax):
        lo, hi = ax.get_ylim()
        return [t.get_window_extent(renderer).y1 + pad
                for pos, t in zip(ax.get_yticks(), ax.get_yticklabels())
                if lo <= pos <= hi and t.get_text()]

    baseline = max(max([ax.title.get_window_extent(renderer).y0] + label_tops(ax))
                   for ax in axes)
    to_fig = fig.transFigure.inverted()
    for ax, letter in zip(axes, letters):
        left = ax.get_tightbbox(renderer).x0
        x, y = to_fig.transform((left, baseline))
        fig.text(x, y, letter, ha="left", va="baseline",
                 fontsize=plt.rcParams["figure.titlesize"], fontweight="bold")


def plot_tsne(ax, data, yscale="log", threads=8):
    """Colour is the implementation, marker shape the approximation."""
    n = [v / 1000 for v in data["n"]]
    original = dict(TOOL_1, label="openTSNE")
    for method, marker, _ in TSNE_METHODS:
        series(ax, n, data[f"{method}_original_{threads}"], dict(original, marker=marker))
        series(ax, n, data[f"{method}_ai_{threads}"], dict(AI, marker=marker))
    style_axes(ax, "openTSNE", "Number of samples [thousands]", yscale)

    implementations = [Line2D([], [], color=style["color"], label=style["label"])
                       for style in (original, AI)]
    methods = [Line2D([], [], color=MUTED, linestyle="none", marker=marker,
                      markeredgewidth=plt.rcParams["lines.linewidth"], label=label)
               for _, marker, label in TSNE_METHODS]
    return implementations + methods


def below(values, top):
    """`values` with everything above `top` set to NaN. Series are drawn with
    clip_on=False, so a point above the y limit would otherwise spill past the
    axes."""
    return [v if v <= top else math.nan for v in values]


def plot_ssgsea(ax, data, cap=1200):
    """Measurements above `cap` seconds stay in the CSV but are not drawn."""
    x = data["gene_sets"]
    series(ax, x, below(data["naive"], cap), NAIVE, label="naive")
    series(ax, x, below(data["gsva"], cap), TOOL_2, label="GSVA")
    series(ax, x, below(data["gseapy"], cap), TOOL_1, label="GSEApy")
    series(ax, x, below(data["ai"], cap), AI)
    style_axes(ax, "ssGSEA", "Number of gene sets")


def plot_graphlets(ax, data):
    x = data["edges_thousands"]
    series(ax, x, data["brute_force"], NAIVE, label="brute force")
    series(ax, x, data["orca"], TOOL_1, label="Orca")
    series(ax, x, data["ai"], AI)
    style_axes(ax, "Graphlets", "Edges [thousands]")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=join(ROOT, "figures", "benchmarks.pdf"),
                    help="output file; the extension picks the format")
    ap.add_argument("--width", type=float, default=6.8425, help="figure width in inches (\\textwidth)")
    ap.add_argument("--height", type=float, default=2.5, help="figure height in inches")
    ap.add_argument("--tsne-yscale", choices=["log", "linear"], default="log",
                    help="y scale of the t-SNE panel; the other panels are always log")
    args = ap.parse_args()

    plt.style.use(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(args.width, args.height))
    # Headroom for panel letters raised above a tick label.
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - 6 / (72 * args.height)))
    key = plot_tsne(axes[0], read_csv("opentsne.csv"), args.tsne_yscale)
    plot_ssgsea(axes[1], read_csv("ssgsea.csv"))
    plot_graphlets(axes[2], read_csv("graphlets.csv"))
    top = axes[0].get_ylim()[1]
    for ax in axes[1:]:
        ax.set_ylim(top=top)

    axes[0].set_ylabel("Time [s]")
    legend_below(axes[0], key, ncol=2)
    legend_below(axes[1], axes[1].get_legend_handles_labels()[0], ncol=2)
    legend_below(axes[2], axes[2].get_legend_handles_labels()[0], ncol=2)

    letter_panels(fig, axes)

    os.makedirs(dirname(abspath(args.out)), exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out} ({args.width} x {args.height} in)")


if __name__ == "__main__":
    main()
