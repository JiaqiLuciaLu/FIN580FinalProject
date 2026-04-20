"""
Plotting functions for AP-Trees replication (Phase E).
Mirrors reference_code/4_Plots/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

from src.code.utils import (
    PY_TREE_GRID_DIR, SUBDIR_3CHAR, KMIN, KMAX, OUTPUT_DIR,
    TREE_GRID_DIR, TREE_PORT_DIR,
)
FEATS_3CHAR = ["LME", "OP", "Investment"]
FEATS_NAMES = ["Size", "Prof", "Inv"]


FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


def _load_sr_n(subdir=SUBDIR_3CHAR):
    """Load SR_N.csv and return (ids, train_sr, valid_sr, test_sr)."""
    path = os.path.join(PY_TREE_GRID_DIR, subdir, "SR_N.csv")
    df = pd.read_csv(path, header=0)
    ids = np.arange(KMIN, KMAX + 1)
    train_sr = df.iloc[0].values.astype(float)
    valid_sr = df.iloc[1].values.astype(float)
    test_sr = df.iloc[2].values.astype(float)
    return ids, train_sr, valid_sr, test_sr


def figure_10a(subdir=SUBDIR_3CHAR, save=True):
    """Figure 10a: Validation SR vs number of portfolios."""
    ids, _, valid_sr, _ = _load_sr_n(subdir)

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.plot(ids, valid_sr, color="red", linewidth=1.5, marker="o", markersize=4)
    ax.set_xlabel("Number of Portfolios", fontsize=24)
    ax.set_ylabel("Validation SR", fontsize=24)
    ax.tick_params(axis="both", labelsize=24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save:
        fig.savefig(
            os.path.join(FIGURE_DIR, f"{subdir}_Validation_gg.png"),
            dpi=150, bbox_inches="tight",
        )
    return fig, ax


def figure_10c(subdir=SUBDIR_3CHAR, save=True):
    """Figure 10c: Testing SR vs number of portfolios."""
    ids, _, _, test_sr = _load_sr_n(subdir)

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.plot(ids, test_sr, color="blue", linewidth=1.5, marker="o", markersize=4)
    ax.set_xlabel("Number of Portfolios", fontsize=24)
    ax.set_ylabel("Testing SR", fontsize=24)
    ax.tick_params(axis="both", labelsize=24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save:
        fig.savefig(
            os.path.join(FIGURE_DIR, f"{subdir}_Testing_gg.png"),
            dpi=150, bbox_inches="tight",
        )
    return fig, ax


# ---- Lambda grid (λ₀, λ₂) parameters matching the full grid ----
LAMBDA0 = np.arange(0, 0.95, 0.05)
LAMBDA2 = 10.0 ** -np.arange(5, 8.25, 0.25)


def _load_sr_grid(kind, port_n=10, subdir=SUBDIR_3CHAR):
    """Load train/valid/test SR grid CSV. Returns (lambda0, lambda2, values)."""
    path = os.path.join(PY_TREE_GRID_DIR, subdir, f"{kind}_SR_{port_n}.csv")
    df = pd.read_csv(path, header=0)
    v = df.values.astype(float)
    return LAMBDA0, LAMBDA2, v


def _plot_lambda_heatmap(kind, label, subdir=SUBDIR_3CHAR, port_n=10,
                         nlev=50, save=True, best_point=None):
    """Contour heatmap of SR over (λ₀, λ₂) grid.
    best_point: (λ₀, λ₂) to mark; defaults to this grid's argmax."""
    l0, l2, v = _load_sr_grid(kind, port_n, subdir)
    x, y = np.meshgrid(l0, l2)

    if best_point is None:
        best_idx = np.unravel_index(np.argmax(v), v.shape)
        best_l0, best_l2 = l0[best_idx[1]], l2[best_idx[0]]
    else:
        best_l0, best_l2 = best_point

    levels = np.linspace(np.min(v), np.max(v), nlev)
    ticks = list(np.unique(np.round(levels, 2)))
    if len(ticks) > 10:
        ticks = ticks[:: len(ticks) // 10]

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.contourf(x, y, v.T, levels=levels)
    ax.plot(best_l0, best_l2, "ro", markersize=8)
    ax.set_xlabel(r"$\lambda_0$", fontsize=20)
    ax.set_ylabel(r"$\lambda_2$", fontsize=20)
    ax.set_yscale("log")
    ax.set_title(f"{subdir}  {label}", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    cbar = fig.colorbar(im, ax=ax, ticks=ticks)
    cbar.ax.set_ylabel("Sharpe Ratio", fontsize=16)
    fig.tight_layout()

    if save:
        fig.savefig(
            os.path.join(FIGURE_DIR, f"{subdir}_{kind}_{port_n}.png"),
            dpi=300, bbox_inches="tight",
        )
    return fig, ax


def figure_10b(subdir=SUBDIR_3CHAR, port_n=10, save=True):
    """Figure 10b: Training SR heatmap over (λ₀, λ₂)."""
    return _plot_lambda_heatmap("train", "Training", subdir, port_n, save=save)


def figure_10d(subdir=SUBDIR_3CHAR, port_n=10, save=True):
    """Figure 10d: Testing SR heatmap over (λ₀, λ₂).
    Red dot = training-best point (matches R reference)."""
    _, _, v_train = _load_sr_grid("train", port_n, subdir)
    best_idx = np.unravel_index(np.argmax(v_train), v_train.shape)
    best_point = (LAMBDA0[best_idx[1]], LAMBDA2[best_idx[0]])
    return _plot_lambda_heatmap("test", "Testing", subdir, port_n, save=save,
                                best_point=best_point)


# ---- Figures 11 & 12: SDF weight heatmaps in characteristic space ----

def _load_all_headers(subdir=SUBDIR_3CHAR):
    """Load column headers from the full filtered portfolio file."""
    path = os.path.join(TREE_PORT_DIR, subdir,
                        "level_all_excess_combined_filtered.csv")
    return list(pd.read_csv(path, nrows=0).columns)


def _load_char_bounds(subdir=SUBDIR_3CHAR, feats=FEATS_3CHAR):
    """Load characteristic min/max arrays for selected portfolios."""
    base = os.path.join(TREE_PORT_DIR, subdir)
    mins, maxs = [], []
    for feat in feats:
        mn = pd.read_csv(os.path.join(base, f"level_all_{feat}_min_filtered.csv"))
        mx = pd.read_csv(os.path.join(base, f"level_all_{feat}_max_filtered.csv"))
        mins.append(mn)
        maxs.append(mx)
    return mins, maxs


def _build_sdf_weight_grid(weights, port_ids, all_headers, char_mins, char_maxs,
                            grid=51, z_slices=None):
    """Build 3D grid of SDF weights in (feat0, feat1, feat2) space.
    Returns (x, y, zs, v) where v has shape (grid, grid, len(zs))."""
    if z_slices is None:
        z_slices = np.linspace(0.1, 0.95, 4)

    matches = [all_headers.index(pid) for pid in port_ids]
    p = len(weights)

    f_min = [df.values[:, matches] for df in char_mins]
    f_max = [df.values[:, matches] for df in char_maxs]

    x = np.linspace(0, 1, grid)
    y = np.linspace(0, 1, grid)
    v = np.zeros((grid, grid, len(z_slices)))

    t = 0
    for i in range(grid):
        for j in range(grid):
            for l in range(len(z_slices)):
                for k in range(p):
                    if (x[j] >= f_min[0][t, k] and x[j] <= f_max[0][t, k] and
                        y[i] >= f_min[1][t, k] and y[i] <= f_max[1][t, k] and
                        z_slices[l] >= f_min[2][t, k] and z_slices[l] <= f_max[2][t, k]):
                        v[i, j, l] += weights[k]
    return x, y, z_slices, v


def _plot_3d_weight_map(x, y, zs, v, feats_names, title, filename,
                         nlev=50, save=True):
    """3D contourf plot of SDF weights at z-slices."""
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "red_white_green",
        [(1, 0, 0), (1, 1, 1), (0, 1, 0)],
        N=nlev - 1,
    )

    v_max = np.max(np.abs(v))
    if v_max > 0:
        v_plot = v / v_max
    else:
        v_plot = v

    xm, ym = np.meshgrid(x, y)
    levels = np.linspace(-1, 1, nlev)

    fig = plt.figure(figsize=(12, 10), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    for k in range(len(zs)):
        ax.contourf(xm, ym, v_plot[:, :, k], offset=zs[k],
                     cmap=cmap, levels=levels)

    ax.set_xlabel(feats_names[0], fontsize=18, labelpad=30)
    ax.set_ylabel(feats_names[1], fontsize=18, labelpad=30)
    ax.set_zlabel(feats_names[2], fontsize=18, labelpad=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_title(title, fontsize=16, pad=20)
    fig.tight_layout()

    if save:
        fig.savefig(os.path.join(FIGURE_DIR, filename),
                     dpi=150, bbox_inches="tight")
    return fig, ax


def figure_11(subdir=SUBDIR_3CHAR, port_n=10, save=True):
    """Figure 11: Combined SDF weight map in characteristic space."""
    grid_dir = os.path.join(TREE_GRID_DIR, subdir)
    weights = pd.read_csv(os.path.join(grid_dir,
                          f"Selected_Ports_Weights_{port_n}.csv")).iloc[:, 0].values
    port_ids = list(pd.read_csv(os.path.join(grid_dir,
                                f"Selected_Ports_{port_n}.csv"), nrows=0).columns)

    all_headers = _load_all_headers(subdir)
    char_mins, char_maxs = _load_char_bounds(subdir)

    x, y, zs, v = _build_sdf_weight_grid(weights, port_ids, all_headers,
                                           char_mins, char_maxs)

    return _plot_3d_weight_map(
        x, y, zs, v, FEATS_NAMES,
        f"SDF Weights — {subdir} (K={port_n})",
        f"{subdir}_SDF_weights_{port_n}.png",
        save=save,
    )


def figure_12(subdir=SUBDIR_3CHAR, port_n=10, save=True):
    """Figure 12: Per-portfolio weight maps in characteristic space."""
    grid_dir = os.path.join(TREE_GRID_DIR, subdir)
    weights = pd.read_csv(os.path.join(grid_dir,
                          f"Selected_Ports_Weights_{port_n}.csv")).iloc[:, 0].values
    port_ids = list(pd.read_csv(os.path.join(grid_dir,
                                f"Selected_Ports_{port_n}.csv"), nrows=0).columns)

    all_headers = _load_all_headers(subdir)
    char_mins, char_maxs = _load_char_bounds(subdir)

    figs = []
    for k in range(len(weights)):
        single_w = np.zeros(len(weights))
        single_w[k] = weights[k]
        x, y, zs, v = _build_sdf_weight_grid(single_w, port_ids, all_headers,
                                               char_mins, char_maxs)
        fig, ax = _plot_3d_weight_map(
            x, y, zs, v, FEATS_NAMES,
            f"Portfolio {port_ids[k]} (w={weights[k]:+.3f})",
            f"{subdir}_port_{k+1}_{port_ids[k].replace('.','_')}.png",
            save=save,
        )
        figs.append(fig)
    return figs
