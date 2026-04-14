"""
SDF time-series regressions against FF factor sets, producing Table 1.

Mirrors `reference_code/3_Metrics_Collection/SDF_TimeSeries_Regressions.R`.

Factor file column layout (tradable_factors.csv, 1-indexed as R reads it):
    1 Date   2 Mkt-RF  3 LME   4 BEME  5 OP  6 Investment
    7 r12_2  8 ST_REV  9 LT_REV 10 AC  11 IdioVol  12 Lturnover  13 rf

R's `factors = c('Date','market',feats_list)` maps 'market' to the Mkt-RF
column (position 2) and `feats_list` = ['LME','BEME','r12_2','OP','Investment',...].
So in *that* mapping:
    match('market')  -> 2
    match('LME')     -> 3
    match('BEME')    -> 4
    match('r12_2')   -> 5
    match('OP')      -> 6
    match('Investment') -> 7
    ...
XSF for our 3-char case (LME, OP, Investment) = cols [2, 3, 6, 7] in the file.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import utils


# Factor set column *indices into tradable_factors.csv* (0-indexed for Python).
# These come straight from SDF_TimeSeries_Regressions.R:14–26.
FF3_COLS = [1, 2, 3]          # Mkt-RF, LME, BEME
FF5_COLS = [1, 2, 3, 5, 6]    # Mkt-RF, LME, BEME, Investment, r12_2  (R: [2,3,4,6,7])
FF11_COLS = list(range(1, 12))  # Mkt-RF ... Lturnover  (R: 2:12)


def _xsf_cols(feat1=utils.FEAT1, feat2=utils.FEAT2, feats_list=utils.FEATS_LIST):
    """
    Reproduce R's XSF option construction:
        factors = c('Date','market',feats_list)
        feats_chosen = c('LME', feats_list[feat1], feats_list[feat2])
        option = match(c('market', feats_chosen), factors)
    In this mapping, 'market' is position 2 of the factors vector, and
    feats_list entries are positions 3..12. Because factors and the file's
    column order coincide here, the returned `option` equals the column
    indices into tradable_factors.csv (1-indexed).

    Returns 0-indexed column indices for Python.
    """
    factors = ["Date", "market"] + list(feats_list)
    feats_chosen = ["market", "LME", feats_list[feat1 - 1], feats_list[feat2 - 1]]
    option_1idx = [factors.index(f) + 1 for f in feats_chosen]
    return [i - 1 for i in option_1idx]


def ff_regression(sdf, factor_mat, option):
    """
    Port of FF_regression(...). `factor_mat` is the test-period slice of
    tradable_factors.csv (T × 13). `option` is either a list of column
    indices (0-indexed) or one of 'FF3', 'FF5', 'FF11'.

    Returns (alpha, SE(alpha), t-stat, p-value).
    """
    if isinstance(option, str):
        cols = {"FF3": FF3_COLS, "FF5": FF5_COLS, "FF11": FF11_COLS}[option]
    else:
        cols = option

    X = factor_mat.iloc[:, cols].to_numpy(dtype=float)
    X = sm.add_constant(X, has_constant="add")
    model = sm.OLS(sdf, X).fit()
    return (
        float(model.params[0]),
        float(model.bse[0]),
        float(model.tvalues[0]),
        float(model.pvalues[0]),
    )


def compute_statistics(sdf, factor_mat, xsf_option):
    """
    Run FF3, FF5, XSF, FF11 regressions and return the 16-value vector
    [α × 4, SE × 4, t-stat × 4, p-value × 4] in that order.
    """
    alpha = np.zeros(4)
    se = np.zeros(4)
    tstat = np.zeros(4)
    pval = np.zeros(4)
    for k, opt in enumerate(["FF3", "FF5", xsf_option, "FF11"]):
        alpha[k], se[k], tstat[k], pval[k] = ff_regression(sdf, factor_mat, opt)
    return np.concatenate([alpha, se, tstat, pval])


def sdf_regression(
    result_dir,
    factor_path,
    port_name="Selected_Ports_10.csv",
    weight_name="Selected_Ports_Weights_10.csv",
    feat1=utils.FEAT1,
    feat2=utils.FEAT2,
    feats_list=utils.FEATS_LIST,
):
    """
    Port of SDF_regression(...). Reads selected ports + weights, constructs
    SDF over the test period [361:636] (1-indexed in R), normalizes to mean 1,
    runs FF3/FF5/XSF/FF11 regressions, writes TimeSeriesAlpha.csv,
    returns DataFrame of results.
    """
    factor_file = os.path.join(factor_path, "tradable_factors.csv")
    # R reads rows 361:636 (1-indexed inclusive) -> Python 360:636
    factor_mat = pd.read_csv(factor_file).iloc[360:636].reset_index(drop=True)

    port_ret = pd.read_csv(os.path.join(result_dir, port_name)).iloc[360:636]
    w = pd.read_csv(os.path.join(result_dir, weight_name)).iloc[:, 0].to_numpy(dtype=float)

    sdf = port_ret.to_numpy(dtype=float) @ w
    sdf = sdf / sdf.mean()

    xsf_cols = _xsf_cols(feat1=feat1, feat2=feat2, feats_list=feats_list)
    result = compute_statistics(sdf, factor_mat, xsf_cols)

    cols = [
        "FF3 Alpha", "FF5 Alpha", "XSF Alpha", "FF11 Alpha",
        "FF3 SE", "FF5 SE", "XSF SE", "FF11 SE",
        "FF3 T-Stat", "FF5 T-Stat", "XSF T-Stat", "FF11 T-Stat",
        "FF3 P-val", "FF5 P-val", "XSF P-val", "FF11 P-val",
    ]
    df = pd.DataFrame([result], columns=cols)
    # R writes to <grid_root>/SDFTests/<subdir>/TimeSeriesAlpha.csv, where
    # <grid_root> is the parent of result_dir (result_dir = grid_root/subdir).
    subdir = utils.char_subdir(feat1, feat2, feats_list)
    grid_root = os.path.dirname(result_dir)
    sdf_dir = os.path.join(grid_root, "SDFTests", subdir)
    os.makedirs(sdf_dir, exist_ok=True)
    df.to_csv(os.path.join(sdf_dir, "TimeSeriesAlpha.csv"), index=False)
    return df
