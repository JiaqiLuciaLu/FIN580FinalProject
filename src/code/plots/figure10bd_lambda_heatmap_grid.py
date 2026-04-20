"""1-1 Python translation of
`reference_code/4_Plots/Figure10bd Lambda Heatmap Grid.ipynb`.

Figure 10b / 10d: contour heatmap of Sharpe Ratio over (λ₀, λ₂) for fixed K,
for the 3-char cross-section LME_OP_Investment. The argmax on the training
grid is marked with a red dot on BOTH train and test heatmaps.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.code import utils


def plot_lambda_heatmap(x, y, v, lambda0_index, lambda2_index,
                        plot_out, filename, dataset, title,
                        p=10, nlev=50, dpi=300):
    levels = np.linspace(np.min(v), np.max(v), nlev)
    ticks = list(np.unique(np.round(levels, 2)))
    if len(ticks) > 10:
        ticks = ticks[0:len(ticks):(len(ticks) // 10)]

    im = plt.contourf(x, y, v, levels=levels)
    plt.plot(lambda0_index, lambda2_index, "ro")
    plt.xlabel(r"$\lambda_0$")
    plt.ylabel(r"$\lambda_2$")
    plt.title(filename + " " + title)
    plt.yscale("log")
    cbar = plt.colorbar(im, ticks=ticks)
    cbar.ax.set_yticklabels(ticks)
    cbar.ax.set_ylabel("Sharpe Ratio")
    plt.savefig(
        os.path.join(plot_out, f"{filename}_{dataset}_{p}.png"),
        dpi=dpi,
    )
    plt.clf()


def run(gridsearchpath=None, plot_out=None, p=10, nlev=50):
    """Replicates the notebook body.

    Parameters
    ----------
    gridsearchpath : str, optional
        Directory containing `<filename>/{train,test}_SR_<p>.csv`.
        Defaults to ``utils.PY_TREE_GRID_DIR``.
    plot_out : str, optional
        Output directory. Defaults to ``<PROJECT_ROOT>/plots/Figure10``.
    p : int
        Number of portfolios K (default 10).
    nlev : int
        Contour levels (default 50).
    """
    if gridsearchpath is None:
        gridsearchpath = utils.PY_TREE_GRID_DIR
    if plot_out is None:
        plot_out = os.path.join(utils.PROJECT_ROOT, "plots", "Figure10")
    os.makedirs(plot_out, exist_ok=True)

    # ipynb: lambda0 = [i*0.05 for i in range(19)]
    #        lambda2 = [0.1**(0.25*x) for x in range(20, 33)]
    lambda0 = np.array([i * 0.05 for i in range(19)])
    lambda2 = np.array([0.1 ** (0.25 * x) for x in range(20, 33)])

    x, y = np.meshgrid(lambda0, lambda2)

    # ipynb: feats = ['LME','BEME','r12_2','OP','Investment','St_Rev',...]
    #        feat1=0, feat2=3, feat3=4   -> 'LME_OP_Investment'
    feats = utils.FEATS_LIST
    feat1, feat2, feat3 = 0, 3, 4
    filename = "_".join([feats[feat1], feats[feat2], feats[feat3]])

    # ---- Train heatmap -----------------------------------------------------
    v = np.array(
        pd.read_csv(
            os.path.join(gridsearchpath, filename, f"train_SR_{p}.csv"),
            header=0,
        )
    ).T
    index = np.unravel_index(np.argmax(v), v.shape)
    plot_lambda_heatmap(
        x, y, v,
        lambda0[index[1]], lambda2[index[0]],
        plot_out, filename,
        dataset="train", title="Training", p=p, nlev=nlev,
    )

    # ---- Test heatmap (red dot kept at training argmax, per notebook) ------
    v = np.array(
        pd.read_csv(
            os.path.join(gridsearchpath, filename, f"test_SR_{p}.csv"),
            header=0,
        )
    ).T
    plot_lambda_heatmap(
        x, y, v,
        lambda0[index[1]], lambda2[index[0]],
        plot_out, filename,
        dataset="test", title="Testing", p=p, nlev=nlev,
    )


if __name__ == "__main__":
    import sys
    gp = sys.argv[1] if len(sys.argv) > 1 else None
    op = sys.argv[2] if len(sys.argv) > 2 else None
    run(gridsearchpath=gp, plot_out=op)
    print(f"wrote Train + Test heatmaps to "
          f"{op or os.path.join(utils.PROJECT_ROOT, 'plots', 'Figure10')}")
