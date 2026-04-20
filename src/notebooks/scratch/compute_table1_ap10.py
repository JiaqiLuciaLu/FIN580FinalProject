"""Compute Table 1, AP-Trees(10) column only.

Uses existing processed outputs — no pipeline rerun needed:
  Selected_Ports_10.csv, Selected_Ports_Weights_10.csv, tradable_factors.csv

Reports (test period Jan 1994 – Dec 2016, rows 361..636 in the R index):
  SDF SR        : monthly Sharpe of SDF = ports @ weights
  α / t-stat    : intercept and t of SDF regressed on 1 + factor set
  XS R²_adj     : per-portfolio TS regressions -> cross-sectional adj R²
for FF3, FF5, XSF, FF11 (matches `reference_code/4_Plots/FigureC8ab_XSR2.R`).
"""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.code import utils


SUBDIR = "LME_OP_Investment"
T0, T1 = 360, 636   # Python 0-based slice == R 361:636 inclusive


# ---- Factor column groups (0-indexed in tradable_factors.csv) --------------
# Header: Date, Mkt-RF, LME, BEME, OP, Investment, r12_2, ST_REV, LT_REV, AC,
#         IdioVol, Lturnover, rf
FF3_COLS  = [1, 2, 3]            # R c(2,3,4)   : Mkt-RF, LME, BEME
FF5_COLS  = [1, 2, 3, 5, 6]      # R c(2,3,4,6,7): +Investment, r12_2
FF11_COLS = list(range(1, 12))   # R 2:12       : Mkt-RF + 10 chars
# XSF for LME_OP_Investment = market + LME + OP + Investment
XSF_COLS  = [1, 2, 4, 5]


def alpha_tstat(y, factors):
    """Regress y on [1, factors]. Return (alpha, t, n_factors)."""
    X = sm.add_constant(factors)
    res = sm.OLS(y, X).fit()
    return float(res.params[0]), float(res.tvalues[0]), factors.shape[1]


def xs_r2_adj(ports, factors):
    """Per-portfolio time-series regressions -> cross-sectional adj R².

    Mirrors `FigureC8ab_XSR2.R::FF_regression` final block:
        rs      = 1 - sum(alphas^2) / sum(avg_ret^2)
        rs_adj  = 1 - (1 - rs) * N / (N - ncol(X))
    where N is number of portfolios and ncol(X) = n_factors + 1 (intercept).
    Returns NaN if N - ncol(X) <= 0.
    """
    X = sm.add_constant(factors)
    N = ports.shape[1]
    alphas = np.empty(N)
    avg_ret = np.empty(N)
    for i in range(N):
        res = sm.OLS(ports[:, i], X).fit()
        alphas[i] = res.params[0]
        avg_ret[i] = ports[:, i].mean()
    rs = 1.0 - (alphas ** 2).sum() / (avg_ret ** 2).sum()
    denom = N - X.shape[1]
    if denom <= 0:
        return float("nan")
    return 1.0 - (1.0 - rs) * N / denom


def main():
    out_dir = os.path.join(utils.PY_TREE_GRID_DIR, SUBDIR)

    ports_full = pd.read_csv(os.path.join(out_dir, "Selected_Ports_10.csv"))
    weights = pd.read_csv(
        os.path.join(out_dir, "Selected_Ports_Weights_10.csv")
    ).iloc[:, 0].to_numpy(dtype=float)

    factors_full = pd.read_csv(os.path.join(utils.FACTOR_DIR, "tradable_factors.csv"))

    ports = ports_full.iloc[T0:T1].to_numpy(dtype=float)         # (276, 10)
    fmat  = factors_full.iloc[T0:T1]                              # (276, 13)
    assert ports.shape == (276, 10), ports.shape

    # ---- SDF ---------------------------------------------------------------
    sdf_raw = ports @ weights  # (276,)
    sdf_sr = float(sdf_raw.mean() / sdf_raw.std(ddof=0))

    # ---- α / t for each factor set (α in raw-return units = % per month) ---
    results = {}
    for name, cols in [("FF3", FF3_COLS), ("FF5", FF5_COLS),
                       ("XSF", XSF_COLS), ("FF11", FF11_COLS)]:
        factors = fmat.iloc[:, cols].to_numpy(dtype=float)
        a, t, _ = alpha_tstat(sdf_raw, factors)
        r2 = xs_r2_adj(ports, factors)
        results[name] = (a, t, r2)

    # ---- Print table -------------------------------------------------------
    print()
    print(f"=== Table 1 — AP-Trees(10), cross-section {SUBDIR} ===")
    print(f"{'':<12}{'our':>16}{'paper':>10}")
    print(f"{'SDF SR':<12}{sdf_sr:>16.2f}{0.65:>10.2f}")

    paper_alpha = {"FF3": (0.94, 10.11), "FF5": (0.81, 8.76),
                   "XSF": (0.81, 8.77),  "FF11": (0.89, 9.12)}
    for name in ["FF3", "FF5", "XSF", "FF11"]:
        a, t, _ = results[name]
        pa, pt = paper_alpha[name]
        print(f"α {name:<9}{f'{a*100:.2f} [{t:.2f}]':>16}{f'{pa:.2f} [{pt:.2f}]':>10}")

    paper_r2 = {"FF3": 0.180, "FF5": 0.110, "XSF": 0.280, "FF11": None}
    for name in ["FF3", "FF5", "XSF", "FF11"]:
        _, _, r = results[name]
        pr = paper_r2[name]
        our_s = "—" if np.isnan(r) else f"{100*r:.1f}%"
        pap_s = "—" if pr is None else f"{100*pr:.1f}%"
        print(f"XS R²adj {name:<3}{our_s:>16}{pap_s:>10}")
    print()


if __name__ == "__main__":
    main()
