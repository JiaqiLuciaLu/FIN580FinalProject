"""Python translation of `reference_code/3_Metrics_Collection/SDF_TimeSeries_Regressions.R`.

Source file to compute the statistics of the portfolios.

The basic input `port_ret` is always a T*N matrix, N is the number of
portfolios and T is the length of time span.
"""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.code import utils


def FF_regression(port_ret, factor_path, option):
    """Fama French regression. First to choose the factor."""
    factor_file = os.path.join(factor_path, "tradable_factors.csv")
    # R: read.table(..., header=T)[361:636, ]  (1-indexed, inclusive)
    factor_mat = pd.read_csv(factor_file).iloc[360:636, :]

    # R's `class(option)=='numeric'` — numeric vector of column indices (1-idx)
    if isinstance(option, (list, tuple, np.ndarray)):
        cols = [int(c) - 1 for c in option]  # R 1-indexed → Py 0-indexed
        factor = factor_mat.iloc[:, cols].to_numpy(dtype=float)
        X = np.column_stack([np.ones(factor.shape[0]), factor])
    elif option == "FF3":
        factor = factor_mat.iloc[:, 1:4].to_numpy(dtype=float)          # R: 2:4
        X = np.column_stack([np.ones(factor.shape[0]), factor])
    elif option == "FF5":
        factor = factor_mat.iloc[:, [1, 2, 3, 5, 6]].to_numpy(dtype=float)  # R: c(2,3,4,6,7)
        X = np.column_stack([np.ones(factor.shape[0]), factor])
    elif option == "FF11":
        factor = factor_mat.iloc[:, 1:12].to_numpy(dtype=float)         # R: 2:12
        X = np.column_stack([np.ones(factor.shape[0]), factor])
    else:
        raise ValueError(f"Unknown option: {option!r}")

    port_ret = np.asarray(port_ret, dtype=float).ravel()

    # R: lm(port_ret ~ X - 1) — no intercept, since X already has a column of 1s.
    model = sm.OLS(port_ret, X).fit()

    # R computes `oos = port_ret - X[,-1] %*% coef(summary(model))[-1,1]`
    # but never uses it; mirrored here and discarded.
    _ = port_ret - X[:, 1:] @ model.params[1:]

    # R: coef(summary(model))[1, 1..4] = Estimate, Std.Error, t-value, Pr(>|t|)
    return np.array([
        model.params[0],
        model.bse[0],
        model.tvalues[0],
        model.pvalues[0],
    ])


def compute_Statistics(port_ret, factor_path, option):
    """Compute: Mean Absolute Alpha, Maximum Sharpe, GRS (as stored in R)."""
    alpha = np.zeros(4)
    se = np.zeros(4)
    tStat = np.zeros(4)
    pval = np.zeros(4)

    # FF3 regression
    res_FF = FF_regression(port_ret, factor_path, "FF3")
    alpha[0], se[0], tStat[0], pval[0] = res_FF
    # FF5 regression
    res_FF = FF_regression(port_ret, factor_path, "FF5")
    alpha[1], se[1], tStat[1], pval[1] = res_FF
    # XSF regression
    res_FF = FF_regression(port_ret, factor_path, option)
    alpha[2], se[2], tStat[2], pval[2] = res_FF
    # FF11 regression
    res_FF = FF_regression(port_ret, factor_path, "FF11")
    alpha[3], se[3], tStat[3], pval[3] = res_FF

    return np.concatenate([alpha, se, tStat, pval])


###################
### Main code   ###
###################


# R declares: results = matrix(nrow = 36, ncol = 16). Module-level, unused by
# SDF_regression; kept as a comment for faithfulness.


def SDF_regression(feats_list=None, feat1=utils.FEAT1, feat2=utils.FEAT2,
                   factor_path=utils.FACTOR_DIR,
                   port_path=utils.PY_TREE_GRID_DIR,
                   port_name="/Selected_Ports_10.csv",
                   weight_name="/Selected_Ports_Weights_10.csv"):
    if feats_list is None:
        feats_list = utils.FEATS_LIST

    factors = ["Date", "market"] + list(feats_list)
    T0 = 361
    T1 = 636

    feats_chosen = ["LME", feats_list[feat1 - 1], feats_list[feat2 - 1]]
    print(feats_chosen)
    # R: match(c('market', feats_chosen), factors) — 1-indexed positions.
    option = [factors.index(name) + 1 for name in (["market"] + feats_chosen)]
    sub_dir = "_".join(feats_chosen)

    # R: paste(port_path, sub_dir, port_name, sep='') where port_name starts with '/'.
    port_ret_full = pd.read_csv(
        os.path.join(port_path, sub_dir, port_name.lstrip("/"))
    )
    port_ret = port_ret_full.iloc[T0 - 1:T1, :].to_numpy(dtype=float)

    w_full = pd.read_csv(
        os.path.join(port_path, sub_dir, weight_name.lstrip("/"))
    )
    w = w_full.iloc[:, 0].to_numpy(dtype=float)

    sdf = port_ret @ w
    sdf = sdf / sdf.mean()

    result = compute_Statistics(sdf, factor_path, option)

    col_names = [
        "FF3 Alpha", "FF5 Alpha", "XSF Alpha", "FF11 Alpha",
        "FF3 SE", "FF5 SE", "XSF SE", "FF11 SE",
        "FF3 T-Stat", "FF5 T-Stat", "XSF T-Stat", "FF11 T-Stat",
        "FF3 P-val", "FF5 P-val", "XSF P-val", "FF11 P-val",
    ]
    result_df = pd.DataFrame([result], columns=col_names)

    os.makedirs(os.path.join(port_path, "SDFTests"), exist_ok=True)
    os.makedirs(os.path.join(port_path, "SDFTests", sub_dir), exist_ok=True)
    result_df.to_csv(
        os.path.join(port_path, "SDFTests", sub_dir, "TimeSeriesAlpha.csv"),
        index=False,
    )
    return result_df


