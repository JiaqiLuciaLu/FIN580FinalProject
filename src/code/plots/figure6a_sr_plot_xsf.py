"""1-1 Python translation of the Figure 6a block in
`reference_code/4_Plots/Figure6a_7_8_SR_Plot_XSF.R` (function `SR_Plot_XSF`,
first plot only).

Reads an `SR_Summary` CSV with 36 rows (one per cross-section) and at least
13 columns. Expected column indices (R 1-based):
    4  -> Id                    (cross-section label)
    5  -> Triple Sort (32) SR
    6  -> Triple Sort (64) SR
    8  -> XSF SR
    10 -> AP-Tree K=10 SR   (sorted ascending for x-axis order)
    11 -> AP-Tree K=40 SR   (used when p=40)

Writes `<plot_path>/SRwithXSF_<p>.png` (p ∈ {10, 40}).
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from src.code import utils


# ggplot default discrete palette for n=4 groups (hcl hues 15/105/195/285).
_GGPLOT_COLORS = ["#F8766D", "#7CAE00", "#00BFC4", "#C77CFF"]
# ggplot default shape scale for n=4: 16, 17, 15, 3 → matplotlib equivalents.
_GGPLOT_SHAPES = ["o", "^", "s", "+"]
# ggplot linetypes in the R script's order: solid, twodash, dotted, longdash.
_LINETYPES = [
    "-",                  # solid
    (0, (3, 2, 1, 2)),    # twodash
    (0, (1, 3)),          # dotted
    (0, (8, 4)),          # longdash
]


def sr_plot_xsf(sr_summary_file, plot_path=None, p=10):
    """Replicates `SR_Plot_XSF(SR_Summary_File, plot_path)` from the R source.

    Parameters
    ----------
    sr_summary_file : str
        Path to CSV with 36 cross-sections × ≥13 columns (see module docstring).
    plot_path : str, optional
        Output directory. Defaults to ``<PROJECT_ROOT>/plots/Figure6``.
    p : int
        10 or 40. Selects the AP-Tree K to sort by and plot.
    """
    if plot_path is None:
        plot_path = os.path.join(utils.PROJECT_ROOT, "plots", "Figure6")
    os.makedirs(plot_path, exist_ok=True)

    # R: read.table(..., header=T, sep=',')[1:36,]
    sr = pd.read_csv(sr_summary_file).iloc[:36].reset_index(drop=True)

    # R uses 1-based column indices. values = sr[order(sr.tree10), c(4,5,6,8,10)]
    # (or c(4,5,6,8,11) for p=40). Columns 0-based: [3, 4, 5, 7, 9] or [3,4,5,7,10].
    sort_col_idx = 9 if p == 10 else 10
    cols = [3, 4, 5, 7, sort_col_idx]
    values = sr.iloc[sr.iloc[:, sort_col_idx].argsort(kind="stable")].iloc[:, cols]
    values.columns = ["Id", "ts32", "ts64", "xsf", "aptree"]

    plot_name = os.path.join(plot_path, f"SRwithXSF_{p}.png")

    # R plot order (legend order): AP-Tree, Triple Sort (32), Triple Sort (64), XSF.
    # XSF series commented out — we don't have XSF SR data for our 36 cross-sections.
    series = [
        ("AP-Tree",          values["aptree"].to_numpy()),
        ("Triple Sort (32)", values["ts32"].to_numpy()),
        ("Triple Sort (64)", values["ts64"].to_numpy()),
        # ("XSF",              values["xsf"].to_numpy()),
    ]

    ids = values["Id"].astype(str).tolist()
    x = range(len(ids))

    text_size = 16
    fig, ax = plt.subplots(figsize=(14.7, 5.7))
    for i, (label, y) in enumerate(series):
        ax.plot(
            x, y,
            linestyle=_LINETYPES[i],
            color=_GGPLOT_COLORS[i],
            linewidth=1.5,
            marker=_GGPLOT_SHAPES[i],
            markersize=8,
            markerfacecolor=_GGPLOT_COLORS[i],
            markeredgecolor=_GGPLOT_COLORS[i],
            label=label,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(ids, rotation=90, fontsize=text_size)
    ax.set_xlabel("Cross-sections", fontsize=text_size)
    ax.set_ylabel("Monthly Sharpe Ratio (SR)", fontsize=text_size)
    ax.tick_params(axis="y", labelsize=text_size)

    # ggplot theme: no grid, no panel bg, only left+bottom axis lines.
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")

    ax.legend(
        title="Basis portfolios:",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=text_size,
        title_fontsize=text_size,
        frameon=False,
    )

    fig.savefig(plot_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_name


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python figure6a_sr_plot_xsf.py <SR_Summary.csv> [plot_path] [p]")
        sys.exit(1)
    sr_file = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    p_val = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    path = sr_plot_xsf(sr_file, out_dir, p=p_val)
    print(f"wrote {path}")
