"""Assemble paper Table 1 from the full-grid outputs (job 3118309).

Four columns: AP-Trees(10), AP-Trees(40), TS(32), TS(64) for cross-section
LME_OP_Investment on the test period (1994-2016).

For each column:
  SDF SR        = mean / std of raw SDF = Selected_Ports @ Weights  (test slice)
  α / t-stat    = intercept and t of raw SDF regressed on [1, factors]
                  (α reported × 100 for % per month, matching paper scale)
  XS R²_adj     = cross-sectional adj R² from per-portfolio TS regressions
                  via `plots.figurec8ab_xsr2.compute_Statistics`
"""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.code import utils
from src.code.plots.figurec8ab_xsr2 import compute_Statistics


SUBDIR = utils.SUBDIR_3CHAR         # LME_OP_Investment
T0, T1 = 361, 636                   # R 1-based inclusive (test period)
T0_PY, T1_PY = T0 - 1, T1           # Python slice [360:636) == 276 months

# Factor-group indices into tradable_factors.csv (1-based, matching R slice
# conventions in figurec8ab_xsr2).
FF3_OPT  = list(range(2, 5))        # R 2:4   -> Mkt-RF, LME, BEME
FF5_OPT  = [2, 3, 4, 6, 7]          # R c(2,3,4,6,7)
FF11_OPT = list(range(2, 13))       # R 2:12

# XSF for LME_OP_Investment: market + LME + OP + Investment.
# The faithful port's automatic match() resolves XSF via Python factors list
# = ['Date','market'] + FEATS_LIST. Because FEATS_LIST has r12_2 before OP,
# the resulting numeric indices collide with CSV column order. We pass the
# XSF option explicitly by CSV position so the Table matches the paper.
XSF_OPT = [2, 3, 5, 6]              # CSV 0-based [1,2,4,5] = Mkt-RF,LME,OP,Investment

PAPER = {
    "AP10": dict(sr=0.65, alpha={"FF3": (0.94, 10.11), "FF5": (0.81, 8.76),
                                  "XSF": (0.81, 8.77),  "FF11": (0.89, 9.12)},
                  r2={"FF3": 0.180, "FF5": 0.110, "XSF": 0.280, "FF11": None}),
    "AP40": dict(sr=0.69, alpha={"FF3": (0.90, 11.03), "FF5": (0.76, 9.60),
                                  "XSF": (0.76, 9.46),  "FF11": (0.80, 9.60)},
                  r2={"FF3": 0.510, "FF5": 0.640, "XSF": 0.650, "FF11": 0.420}),
    "TS32": dict(sr=0.51, alpha={"FF3": (0.75, 7.40),  "FF5": (0.47, 5.57),
                                  "XSF": (0.46, 5.39),  "FF11": (0.37, 4.29)},
                  r2={"FF3": 0.820, "FF5": 0.910, "XSF": 0.910, "FF11": 0.920}),
    "TS64": dict(sr=0.53, alpha={"FF3": (0.84, 8.13),  "FF5": (0.61, 6.73),
                                  "XSF": (0.61, 6.69),  "FF11": (0.65, 6.91)},
                  r2={"FF3": 0.820, "FF5": 0.900, "XSF": 0.900, "FF11": 0.870}),
}


def load_factors_slice(cols):
    """Load tradable_factors.csv test slice with R 1-based column indices."""
    mat = pd.read_csv(os.path.join(utils.FACTOR_DIR, "tradable_factors.csv"))
    py_cols = [c - 1 for c in cols]
    return mat.iloc[T0_PY:T1_PY, py_cols].to_numpy(dtype=float)


def sdf_alpha(sdf_raw, factors):
    """Return (alpha, t) of OLS(sdf_raw ~ 1 + factors)."""
    X = sm.add_constant(factors)
    res = sm.OLS(sdf_raw, X).fit()
    return float(res.params[0]), float(res.tvalues[0])


def col_stats(grid_dir, K):
    """All Table 1 stats for one column.

    Returns dict with 'sr', 'alpha': {name:(α,t)}, 'r2': {name: adj_R²}.
    """
    sd = os.path.join(grid_dir, SUBDIR)
    ports_full = pd.read_csv(os.path.join(sd, f"Selected_Ports_{K}.csv"))
    weights = pd.read_csv(
        os.path.join(sd, f"Selected_Ports_Weights_{K}.csv")
    ).iloc[:, 0].to_numpy(dtype=float)

    ports_full_np = ports_full.to_numpy(dtype=float)          # (636, K)
    ports_test = ports_full_np[T0_PY:T1_PY, :]                # (276, K)

    # SDF on test period — raw units (% per month when ×100).
    sdf_raw = ports_test @ weights
    sr = float(sdf_raw.mean() / sdf_raw.std(ddof=0))

    alpha_d = {}
    for name, cols in [("FF3", FF3_OPT), ("FF5", FF5_OPT),
                       ("XSF", XSF_OPT), ("FF11", FF11_OPT)]:
        a, t = sdf_alpha(sdf_raw, load_factors_slice(cols))
        alpha_d[name] = (a * 100.0, t)

    # XS R² via faithful compute_Statistics — pass FULL panel (it slices T0:T1).
    # stats_vec layout: [mean|α| × 4, cs_rsq × 4, cs_adjrsq × 4, cs_r × 4]
    _, stats_vec, *_ = compute_Statistics(
        ports_full_np, utils.FACTOR_DIR, XSF_OPT, T0, T1
    )
    adj = stats_vec[8:12]
    r2_d = {"FF3": adj[0], "FF5": adj[1], "XSF": adj[2], "FF11": adj[3]}

    return {"sr": sr, "alpha": alpha_d, "r2": r2_d}


def main():
    cols = {
        "AP10": col_stats(utils.PY_TREE_GRID_DIR, 10),
        "AP40": col_stats(utils.PY_TREE_GRID_DIR, 40),
        "TS32": col_stats(utils.PY_TS_GRID_DIR,   32),
        "TS64": col_stats(utils.PY_TS64_GRID_DIR, 64),
    }

    labels = ["AP10", "AP40", "TS32", "TS64"]
    label_w = 24

    def row(label, vals, paper):
        our = "  ".join(f"{v:>14}" for v in vals)
        pp = "  ".join(f"{p:>14}" for p in paper)
        return f"{label:<{label_w}}{our}   |  paper:  {pp}"

    print()
    print("=== Paper Table 1 — cross-section LME_OP_Investment ===")
    head = "".join(f"{x:>14}  " for x in labels)
    print(f"{'':<{label_w}}{head.strip()}   |  paper: same headers")

    print(row("SDF SR",
              [f"{cols[l]['sr']:.2f}" for l in labels],
              [f"{PAPER[l]['sr']:.2f}" for l in labels]))

    for grp in ["FF3", "FF5", "XSF", "FF11"]:
        print(row(f"α {grp}",
                  [f"{cols[l]['alpha'][grp][0]:.2f} [{cols[l]['alpha'][grp][1]:.2f}]"
                   for l in labels],
                  [f"{PAPER[l]['alpha'][grp][0]:.2f} [{PAPER[l]['alpha'][grp][1]:.2f}]"
                   for l in labels]))

    for grp in ["FF3", "FF5", "XSF", "FF11"]:
        def fmt_our(r):
            return "—" if (r is None or np.isnan(r)) else f"{100*r:.1f}%"
        def fmt_pap(r):
            return "—" if r is None else f"{100*r:.1f}%"
        print(row(f"XS R² adj {grp}",
                  [fmt_our(cols[l]['r2'][grp]) for l in labels],
                  [fmt_pap(PAPER[l]['r2'][grp]) for l in labels]))

    # Save comparison CSV (ours || paper) under plots/table1/.
    def ours_cell(l, metric, grp=None):
        if metric == "sr":
            return f"{cols[l]['sr']:.3f}"
        if metric == "alpha":
            a, t = cols[l]["alpha"][grp]
            return f"{a:.3f} [{t:.3f}]"
        if metric == "r2":
            r = cols[l]["r2"][grp]
            return "" if np.isnan(r) else f"{r:.4f}"

    def paper_cell(l, metric, grp=None):
        if metric == "sr":
            return f"{PAPER[l]['sr']:.2f}"
        if metric == "alpha":
            a, t = PAPER[l]["alpha"][grp]
            return f"{a:.2f} [{t:.2f}]"
        if metric == "r2":
            r = PAPER[l]["r2"][grp]
            return "" if r is None else f"{r:.2f}"

    header_cols = ["metric"]
    for l in labels:
        header_cols += [f"{l}_ours", f"{l}_paper"]

    def build_row(metric_label, metric, grp=None):
        row = [metric_label]
        for l in labels:
            row += [ours_cell(l, metric, grp), paper_cell(l, metric, grp)]
        return row

    rows = [build_row("SDF SR", "sr")]
    for grp in ["FF3", "FF5", "XSF", "FF11"]:
        rows.append(build_row(f"alpha_{grp}", "alpha", grp))
    for grp in ["FF3", "FF5", "XSF", "FF11"]:
        rows.append(build_row(f"xs_r2_adj_{grp}", "r2", grp))

    out = pd.DataFrame(rows, columns=header_cols)
    plots_dir = os.path.join(utils.PROJECT_ROOT, "plots", "table1")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, "table1_comparison.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
