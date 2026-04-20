"""Python translation of
`reference_code/4_Plots/Figure10ac_SDF_SR_N.R` (function `SRN_Plot`).

Figure 10a (Validation SR vs K) and Figure 10c (Testing SR vs K) for the
3-characteristic cross-section LME_OP_Investment (feat1=4, feat2=5).

Fixes a bug in the R source: the original used `sr[2, ]` (validation row) for
both plots. Here the Testing plot uses `sr[3, ]` (testing row).
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from src.code import utils


def srn_plot(grid_search_path, plot_path=None):
    """Replicates `SRN_Plot(GridSearchPath, plot_path)` from the R source.

    Parameters
    ----------
    grid_search_path : str
        Directory containing `<filename>/SR_N.csv`. Mirrors R's `GridSearchPath`.
    plot_path : str, optional
        Output directory. Defaults to ``<PROJECT_ROOT>/plots/Figure10``.
    """
    # R: graph.width <- 14; graph.height <- 9; text.size <- 24
    graph_width = 14
    graph_height = 9
    text_size = 24

    # R: ids = 5:50
    ids = list(range(5, 51))

    # R: feat1 = 4; feat2 = 5
    # filename = paste(c('LME', feats_list[feat1], feats_list[feat2]), collapse='_')
    feat1 = 4
    feat2 = 5
    feats_list = utils.FEATS_LIST
    filename = "_".join(["LME", feats_list[feat1 - 1], feats_list[feat2 - 1]])

    if plot_path is None:
        plot_path = os.path.join(utils.PROJECT_ROOT, "plots", "Figure10")
    os.makedirs(plot_path, exist_ok=True)

    # R: sr_path = paste(GridSearchPath, filename, '/SR_N.csv', sep='')
    sr_path = os.path.join(grid_search_path, filename, "SR_N.csv")
    # R: sr = read.table(sr_path, header=T, sep=',')
    sr = pd.read_csv(sr_path, header=0)

    # SR_N.csv rows: 0=train, 1=validation, 2=testing.
    valid_sr = sr.iloc[1].to_numpy(dtype=float)
    test_sr = sr.iloc[2].to_numpy(dtype=float)

    # ---- Plot 1: Validation ------------------------------------------------
    plot_name = os.path.join(plot_path, f"{filename}_Validation_gg.png")
    _gg_lineplot(
        ids, valid_sr,
        color="red",
        ylabel="Validation SR",
        text_size=text_size,
        graph_width=graph_width,
        graph_height=graph_height,
        out=plot_name,
    )

    # ---- Plot 2: Testing ---------------------------------------------------
    # Fixes R bug where both plots used sr[2,] (the validation row).
    plot_name = os.path.join(plot_path, f"{filename}_Testing_gg.png")
    _gg_lineplot(
        ids, test_sr,
        color="blue",
        ylabel="Testing SR",
        text_size=text_size,
        graph_width=graph_width,
        graph_height=graph_height,
        out=plot_name,
    )


def _gg_lineplot(ids, srs, color, ylabel, text_size, graph_width, graph_height, out):
    """Matplotlib mimic of the ggplot theme used in the R source."""
    fig, ax = plt.subplots(figsize=(graph_width, graph_height))
    # R: geom_line(size=1.5) + geom_point()
    ax.plot(ids, srs, color=color, linewidth=1.5)
    ax.scatter(ids, srs, color=color, zorder=3)

    ax.set_xlabel("Number of Portfolios", fontsize=text_size)
    ax.set_ylabel(ylabel, fontsize=text_size)
    ax.tick_params(axis="both", labelsize=text_size)

    # R theme: no panel grid, no panel background, black axis lines only.
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("black")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        # Default: our Python pipeline output dir.
        grid_path = utils.PY_TREE_GRID_DIR
    else:
        grid_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    srn_plot(grid_path, out_dir)
    print(f"wrote Validation + Testing plots to "
          f"{out_dir or os.path.join(utils.PROJECT_ROOT, 'plots', 'Figure10')}")
