"""Python translation of
`reference_code/4_Plots/Figure1bc Empirical Portfolio Bounding Box.ipynb`.

Two plots for a 2-characteristic cross-section:
  figure_1b_tree — draws the bounding box of every tree leaf at t=0, colored
                   by root-split (first 4 characters of the permutation id).
  figure_1c_ds   — draws the evenly-spaced double-sort grid (4x4) for
                   comparison.

Inputs (for figure_1b_tree) come from R's `combinetrees` output for the
2-char tree: `level_all_{feat}_min.csv` and `level_all_{feat}_max.csv`.
"""

import os

import matplotlib.collections as mc
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd

from src.code import utils


def figure_1b_tree(feats=("LME", "OP"),
                   tree_port_path=utils.PY_TREE_PORT_DIR,
                   plot_out=None):
    """Tree-portfolio bounding-box plot at t=0 for a 2-char cross-section."""
    if plot_out is None:
        plot_out = os.path.join(utils.PY_PLOTS_DIR, "Conditional_cutoffs")
    os.makedirs(plot_out, exist_ok=True)

    feats = list(feats)
    filename = "_".join(feats)
    feats_min = [
        np.array(pd.read_csv(
            os.path.join(tree_port_path, filename, f"level_all_{feat}_min.csv"),
            header=0,
        ))
        for feat in feats
    ]
    feats_max = [
        np.array(pd.read_csv(
            os.path.join(tree_port_path, filename, f"level_all_{feat}_max.csv"),
            header=0,
        ))
        for feat in feats
    ]

    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    lines = []
    c = []
    for i in range(16):
        for j in range(16):
            k = i * 16 + j
            lines.append([(feats_min[0][0, k], feats_min[1][0, k]),
                          (feats_min[0][0, k], feats_max[1][0, k])])
            c.append(list(colors.items())[i][0])
            lines.append([(feats_min[0][0, k], feats_min[1][0, k]),
                          (feats_max[0][0, k], feats_min[1][0, k])])
            c.append(list(colors.items())[i][0])
            lines.append([(feats_max[0][0, k], feats_min[1][0, k]),
                          (feats_max[0][0, k], feats_max[1][0, k])])
            c.append(list(colors.items())[i][0])
            lines.append([(feats_min[0][0, k], feats_max[1][0, k]),
                          (feats_max[0][0, k], feats_max[1][0, k])])
            c.append(list(colors.items())[i][0])

    lc = mc.LineCollection(lines, color=c, linewidths=1, linestyle="dashed")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_collection(lc)
    ax.autoscale()
    plt.xlabel(feats[0])
    plt.ylabel(feats[1])

    plt.savefig(os.path.join(plot_out, filename + "_tree.png"), dpi=300)
    plt.clf()


def figure_1c_ds(feats=("LME", "OP"), plot_out=None):
    """Double-sort grid (4x4) bounding-box plot."""
    if plot_out is None:
        plot_out = os.path.join(utils.PY_PLOTS_DIR, "Conditional_cutoffs")
    os.makedirs(plot_out, exist_ok=True)

    feats = list(feats)
    filename = "_".join(feats)

    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    lines = []
    c = []

    i = 1
    lines.append([(0.25, 0), (0.25, 1)])
    c.append(list(colors.items())[i][0])
    lines.append([(0.5, 0), (0.5, 1)])
    c.append(list(colors.items())[i][0])
    lines.append([(0.75, 0), (0.75, 1)])
    c.append(list(colors.items())[i][0])

    i = 2
    lines.append([(0, 0.25), (1, 0.25)])
    c.append(list(colors.items())[i][0])
    lines.append([(0, 0.5), (1, 0.5)])
    c.append(list(colors.items())[i][0])
    lines.append([(0, 0.75), (1, 0.75)])
    c.append(list(colors.items())[i][0])

    lc = mc.LineCollection(lines, color=c, linewidths=1, linestyle="dashed")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_collection(lc)
    ax.autoscale()
    plt.xlabel(feats[0])
    plt.ylabel(feats[1])

    plt.savefig(os.path.join(plot_out, filename + "_ds.png"), dpi=300)
    plt.clf()


if __name__ == "__main__":
    figure_1b_tree()
    figure_1c_ds()
