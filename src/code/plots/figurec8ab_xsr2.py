"""1-1 Python translation of
`reference_code/4_Plots/FigureC8ab_XSR2.R`.

Provides:
  FF_regression(port_ret, factor_path, option, T0, T1)
    -> (coeff, se, t, p, rs, adj_rs, rsq)
  compute_Statistics(port_ret, factor_path, option, T0, T1)
    -> (alphas_list, stats_vec, ses_list, ts_list, ps_list,
        rs_list, adj_rs_list)
  XSR2(feats_list, feat1, feat2, factor_path, port_path, port_name,
       plot_path_base, port_type)

Notes on mapping R -> Python:
  • R uses 1-based inclusive slices `[T0:T1, ]`. Python uses
    0-based half-open slices `[T0-1:T1, :]`.
  • `class(option)=='numeric'` (R) is matched by `option` being a list /
    tuple / ndarray of integer column indices (1-based like R).
  • `lm(port_ret ~ X - 1)` fits no extra intercept because X already has a
    column of 1s — we do the same with statsmodels OLS (no add_constant).
  • rsq[1] = cross-sectional R²:  1 - Σα² / Σ avg_ret²
    rsq[2] = adj cross-sectional R²:  1 - (1-rs) * N / (N - ncol(X))
             (NaN when N <= ncol(X))
    rsq[3] = residual-based:  1 - Σε² / Σ port_ret²

Boxplot-generation inside XSR2() is temporarily commented out — compute /
CSV-writing portions remain functional.
"""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

# import matplotlib.pyplot as plt  # boxplot block commented out below


def FF_regression(port_ret, factor_path, option, T0, T1):
    """Fama-French time-series regressions, one per portfolio column.

    Parameters
    ----------
    port_ret : array-like, shape (T, N)
        Full panel of portfolio returns (T months, N portfolios).
    factor_path : str
        Directory containing `tradable_factors.csv`.
    option : int-vector or str
        Either 1-based column indices into tradable_factors (emulates R's
        `class(option)=='numeric'`), or one of 'FF3' / 'FF5' / 'FF11'.
    T0, T1 : int
        R-style 1-based inclusive month indices (e.g. T0=361, T1=636).
    """
    factor_file = os.path.join(factor_path, "tradable_factors.csv")
    factor_mat = pd.read_csv(factor_file)

    # R: factor_mat[T0:T1, ...] is 1-based inclusive -> Python [T0-1:T1]
    if isinstance(option, (list, tuple, np.ndarray)):
        cols = [int(c) - 1 for c in option]  # R 1-based -> Py 0-based
        factor = factor_mat.iloc[T0 - 1:T1, cols].to_numpy(dtype=float)
    elif option == "FF3":
        factor = factor_mat.iloc[T0 - 1:T1, 1:4].to_numpy(dtype=float)             # R: 2:4
    elif option == "FF5":
        factor = factor_mat.iloc[T0 - 1:T1, [1, 2, 3, 5, 6]].to_numpy(dtype=float) # R: c(2,3,4,6,7)
    elif option == "FF11":
        factor = factor_mat.iloc[T0 - 1:T1, 1:12].to_numpy(dtype=float)            # R: 2:12
    else:
        raise ValueError(f"Unknown option: {option!r}")

    X = np.column_stack([np.ones(factor.shape[0]), factor])  # (T_slice, k+1)

    port_ret = np.asarray(port_ret, dtype=float)
    y = port_ret[T0 - 1:T1, :]  # (T_slice, N)
    T_slice, N = y.shape
    kp1 = X.shape[1]

    coeff = np.zeros((kp1, N))
    se    = np.zeros((kp1, N))
    tvals = np.zeros((kp1, N))
    pvals = np.zeros((kp1, N))
    rs_   = np.zeros(N)
    adj_rs = np.zeros(N)

    alphas = np.zeros(N)
    avg_ret = np.zeros(N)
    eps = np.zeros_like(y)
    betas = np.zeros((N, kp1 - 1))

    for i in range(N):
        res = sm.OLS(y[:, i], X).fit()
        coeff[:, i] = res.params
        se[:, i]    = res.bse
        tvals[:, i] = res.tvalues
        pvals[:, i] = res.pvalues
        rs_[i]      = res.rsquared
        adj_rs[i]   = res.rsquared_adj

        alphas[i] = res.params[0]
        betas[i, :] = res.params[1:]
        avg_ret[i] = y[:, i].mean()
        eps[:, i] = y[:, i] - X[:, 1:] @ res.params[1:]

    # R block:
    #   rs = 1 - sum(alphas^2) / sum(avg_ret^2)
    #   rsq = c(rs,
    #           1 - (1-rs) * length(pred_ret) / (length(pred_ret) - ncol(X)),
    #           1 - sum(eps^2)/sum(port_ret[T0:T1,]^2))
    rs_cs = 1.0 - (alphas ** 2).sum() / (avg_ret ** 2).sum()
    # N <= ncol(X) => adj R² is undefined (paper shows "—" for FF11 at K=10).
    denom = N - kp1
    rs_cs_adj = 1.0 - (1.0 - rs_cs) * N / denom if denom > 0 else np.nan
    rs_resid = 1.0 - (eps ** 2).sum() / (y ** 2).sum()
    rsq = np.array([rs_cs, rs_cs_adj, rs_resid])

    return coeff, se, tvals, pvals, rs_, adj_rs, rsq


def compute_Statistics(port_ret, factor_path, option, T0, T1):
    """Run FF_regression for FF3, FF5, `option` (XSF), FF11.

    Returns a 7-tuple mirroring the R list layout:
      (alphas_list, stats_vec, ses_list, ts_list, ps_list, rs_list, adj_rs_list)
    where *_list has 4 entries (one per factor group) and stats_vec is a
    length-16 concatenation of (mean_alpha, cs_rsq, cs_adjrsq, cs_r) each × 4.
    """
    mean_alpha = np.zeros(4)
    cs_rsq = np.zeros(4)
    cs_adjrsq = np.zeros(4)
    cs_r = np.zeros(4)

    groups = ["FF3", "FF5", option, "FF11"]
    alphas_list, ses_list, ts_list, ps_list = [], [], [], []
    rs_list, adj_rs_list = [], []

    for idx, grp in enumerate(groups):
        coeff, se, tvals, pvals, rs_, adj_rs, rsq = FF_regression(
            port_ret, factor_path, grp, T0, T1
        )
        alpha_row = coeff[0, :]
        se_row    = se[0, :]
        t_row     = tvals[0, :]
        p_row     = pvals[0, :]

        mean_alpha[idx] = np.mean(np.abs(alpha_row))
        cs_rsq[idx]    = rsq[0]
        cs_adjrsq[idx] = rsq[1]
        cs_r[idx]      = rsq[2]

        alphas_list.append(alpha_row)
        ses_list.append(se_row)
        ts_list.append(t_row)
        ps_list.append(p_row)
        rs_list.append(rs_)
        adj_rs_list.append(adj_rs)

    stats_vec = np.concatenate([mean_alpha, cs_rsq, cs_adjrsq, cs_r])

    return alphas_list, stats_vec, ses_list, ts_list, ps_list, rs_list, adj_rs_list


def XSR2(feats_list, feat1, feat2, factor_path, port_path, port_name,
         plot_path_base, port_type):
    """1-1 port of the R `XSR2(...)` driver.

    Writes:
      <port_path>/XSR2Tests/<sub_dir>/alpha.csv
      <port_path>/XSR2Tests/<sub_dir>/R2.csv

    Boxplot-generation (per-portfolio pricing error boxplots for FF3/FF5/XSF/
    FF11) is COMMENTED OUT in this port — restore the block below to mirror
    the R ggplot output.
    """
    factors = ["Date", "market"] + list(feats_list)
    T0 = 361
    T1 = 636

    # R: feat1/feat2 are 1-based; feats_list[feat1] picks that element directly.
    feats_chosen = ["LME", feats_list[feat1 - 1], feats_list[feat2 - 1]]
    print(feats_chosen)

    # R: match(c('market', feats_chosen), factors) -> 1-based positions.
    option = [factors.index(name) + 1 for name in (["market"] + feats_chosen)]
    sub_dir = "_".join(feats_chosen)

    port_ret = pd.read_csv(
        os.path.join(port_path, sub_dir, port_name.lstrip("/"))
    ).to_numpy(dtype=float)

    result = compute_Statistics(port_ret, factor_path, option, T0, T1)
    alphas_list, stats_vec, ses_list, ts_list, ps_list, _, _ = result

    # 4 × N matrices (R: alphas = rbind(result[[1]][[1..4]]))
    alphas = np.vstack(alphas_list)
    ses    = np.vstack(ses_list)
    ts     = np.vstack(ts_list)
    ps     = np.vstack(ps_list)

    # Column names (R: cnames depend on port_type)
    if port_type == "ts32":
        cnames = [
            "111", "112", "113", "114", "121", "122", "123", "124",
            "131", "132", "133", "134", "141", "142", "143", "144",
            "211", "212", "213", "214", "221", "222", "223", "224",
            "231", "232", "233", "234", "241", "242", "243", "244",
        ]
    elif port_type == "ts64":
        cnames = [
            "111", "112", "113", "114", "121", "122", "123", "124",
            "131", "132", "133", "134", "141", "142", "143", "144",
            "211", "212", "213", "214", "221", "222", "223", "224",
            "231", "232", "233", "234", "241", "242", "243", "244",
            "311", "312", "313", "314", "321", "322", "323", "324",
            "331", "332", "333", "334", "341", "342", "343", "344",
            "411", "412", "413", "414", "421", "422", "423", "424",
            "431", "432", "433", "434", "441", "442", "443", "444",
        ]
    else:
        # R: cnames = colnames(port_ret); cnames = substring(cnames, 2, nchar(cnames))
        original = pd.read_csv(
            os.path.join(port_path, sub_dir, port_name.lstrip("/"))
        ).columns.tolist()
        cnames = [c[1:] for c in original]

    # -----------------------------------------------------------------------
    # [COMMENTED OUT] per-factor pricing-error boxplot panels (Figure C8).
    # The R block builds a ggplot with `geom_boxplot(stat="identity")` plus
    # three-sigma dotted bands and a zero reference line, saved as:
    #   <plot_path_base>/<port_type>/TimeSeriesAlpha/<sub_dir>/TimeSeriesAlpha_{FF3,FF5,XSF,FF11}.png
    # Restore in a follow-up PR when Figure C8 is needed.
    #
    # plot_path = os.path.join(plot_path_base, port_type, "TimeSeriesAlpha", sub_dir)
    # os.makedirs(plot_path, exist_ok=True)
    # factor_group_names = ["FF3", "FF5", "XSF", "FF11"]
    # text_size = 15
    # for i in range(4):
    #     plot_name = os.path.join(plot_path, f"TimeSeriesAlpha_{factor_group_names[i]}.png")
    #     fig, ax = plt.subplots(figsize=(16, 6))
    #     ... # boxplot + ±3σ dotted band + y=0 red line
    #     fig.savefig(plot_name); plt.close(fig)
    # -----------------------------------------------------------------------

    # R: alphas = rbind(alphas, ses, ts, ps)  -> (16, N)
    alphas_stacked = np.vstack([alphas, ses, ts, ps])

    stats_path = os.path.join(port_path, "XSR2Tests", sub_dir)
    os.makedirs(stats_path, exist_ok=True)

    pd.DataFrame(alphas_stacked, columns=cnames).to_csv(
        os.path.join(stats_path, "alpha.csv"), index=False,
    )
    pd.DataFrame(stats_vec.reshape(-1, 1)).to_csv(
        os.path.join(stats_path, "R2.csv"), index=False,
    )
