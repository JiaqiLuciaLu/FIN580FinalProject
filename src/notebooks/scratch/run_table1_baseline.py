"""
Paper Table 1 baseline replication (columns 1-2: AP-Trees K=10 and K=40).

Runs full (λ₀, λ₂) grid on Phase F clean-CRSP tree portfolios for
LME_OP_Investment, then for each K ∈ {10, 40}:
  - CV-picks best (λ₀, λ₂) via pick_best_lambda
  - Builds SDF = Selected_Ports_K @ Weights on test period
  - Computes SDF Sharpe, α + t-stat vs FF3/FF5/XSF/FF11, XS R²_adj

Target values (paper Table 1):
                    AP-Trees(10)  AP-Trees(40)
  SDF SR            0.65          0.69
  α FF3             0.94 [10.11]  0.90 [11.03]
  α FF5             0.81 [8.76]   0.76 [9.60]
  α XSF             0.81 [8.77]   0.76 [9.46]
  α FF11            0.89 [9.12]   0.80 [9.60]
  XS R²_adj FF3     18.0%         51.0%
  XS R²_adj FF5     11.0%         64.0%
  XS R²_adj XSF     28.0%         65.0%
  XS R²_adj FF11    —             42.0%
"""

import argparse
import os
import time
from typing import Tuple

import numpy as np
import pandas as pd

from src.code import utils
from src.code.portfolio_creation import data_prep
from src.code.ap_pruning import pruning
from src.code.metrics import metrics
from src.code.metrics.regressions import (
    FF3_COLS, FF5_COLS, FF11_COLS, _xsf_cols, ff_regression,
)
from src.code.metrics.xsr2 import xs_r2


LAMBDA0_LIST = np.arange(0.0, 0.95, 0.05).tolist()
LAMBDA2_LIST = (10 ** -np.arange(5.0, 8.25, 0.25)).tolist()
SUBDIR = utils.SUBDIR_3CHAR  # LME_OP_Investment


def grid_exists(out_dir: str) -> bool:
    for cv in ["cv_1", "cv_2", "cv_3", "full"]:
        for i in range(1, len(LAMBDA0_LIST) + 1):
            for j in range(1, len(LAMBDA2_LIST) + 1):
                if not os.path.isfile(os.path.join(out_dir, f"results_{cv}_l0_{i}_l2_{j}.csv")):
                    return False
    return True


def sdf_and_sharpe(selected_path: str, weights_path: str) -> Tuple[np.ndarray, float]:
    """Load Selected_Ports_K + Weights, slice test period, return (sdf_normalized, sr)."""
    port = pd.read_csv(selected_path).iloc[360:636].reset_index(drop=True)
    w = pd.read_csv(weights_path).iloc[:, 0].to_numpy(dtype=float)
    sdf_raw = port.to_numpy(dtype=float) @ w
    sdf = sdf_raw / sdf_raw.mean()  # matches regressions.py normalization
    sr = float(sdf_raw.mean() / sdf_raw.std(ddof=0))
    return sdf, sr


def one_column(out_dir: str, port_path: str, K: int, factor_mat: pd.DataFrame) -> dict:
    """Compute all Table 1 stats for one AP-Trees(K) column."""
    t0 = time.time()
    metrics.pick_best_lambda(
        result_dir=out_dir,
        portfolio_path=port_path,
        portN=K,
        lambda0_list=LAMBDA0_LIST,
        lambda2_list=LAMBDA2_LIST,
        full_cv=False,
        write=True,
    )
    print(f"[K={K}] pick_best_lambda: {time.time()-t0:.1f}s", flush=True)

    selected_path = os.path.join(out_dir, f"Selected_Ports_{K}.csv")
    weights_path = os.path.join(out_dir, f"Selected_Ports_Weights_{K}.csv")
    sdf, sdf_sr = sdf_and_sharpe(selected_path, weights_path)

    xsf_cols = _xsf_cols(feat1=utils.FEAT1, feat2=utils.FEAT2, feats_list=utils.FEATS_LIST)
    alphas = {}
    for name, cols in [("FF3", FF3_COLS), ("FF5", FF5_COLS), ("XSF", xsf_cols), ("FF11", FF11_COLS)]:
        alpha, se, t, p = ff_regression(sdf, factor_mat, cols)
        alphas[name] = (alpha, t)

    selected_test = pd.read_csv(selected_path).iloc[360:636].reset_index(drop=True)
    xsr2_vals = {}
    for name, cols in [("FF3", FF3_COLS), ("FF5", FF5_COLS), ("XSF", xsf_cols), ("FF11", FF11_COLS)]:
        r = xs_r2(selected_test, factor_mat, cols)
        xsr2_vals[name] = r.xs_r2_adj

    return {"K": K, "sdf_sr": sdf_sr, "alphas": alphas, "xsr2_adj": xsr2_vals}


def format_table1(col_10: dict, col_40: dict) -> pd.DataFrame:
    """Assemble paper-layout 2-column table."""
    rows = [("SDF SR", f"{col_10['sdf_sr']:.2f}", f"{col_40['sdf_sr']:.2f}")]
    for factor in ["FF3", "FF5", "XSF", "FF11"]:
        a10, t10 = col_10["alphas"][factor]
        a40, t40 = col_40["alphas"][factor]
        rows.append((
            f"α {factor}",
            f"{a10:.2f} [{t10:.2f}]",
            f"{a40:.2f} [{t40:.2f}]",
        ))
    for factor in ["FF3", "FF5", "XSF", "FF11"]:
        r10 = col_10["xsr2_adj"][factor]
        r40 = col_40["xsr2_adj"][factor]
        rows.append((
            f"XS R²_adj {factor}",
            "—" if np.isnan(r10) else f"{100*r10:.1f}%",
            "—" if np.isnan(r40) else f"{100*r40:.1f}%",
        ))
    return pd.DataFrame(rows, columns=["metric", "AP-Trees (10)", "AP-Trees (40)"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-grid", action="store_true")
    args = ap.parse_args()

    out_dir = os.path.join(utils.PY_TREE_GRID_DIR, SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    port_path = os.path.join(utils.PY_TREE_PORT_DIR, SUBDIR,
                             "level_all_excess_combined_filtered.csv")

    print(f"[table1] Loading ports from {port_path}", flush=True)
    ports = data_prep.load_filtered_tree_portfolios(SUBDIR, tree_port_dir=utils.PY_TREE_PORT_DIR)
    print(f"[table1] Ports shape: {ports.shape}", flush=True)

    if args.skip_grid or grid_exists(out_dir):
        print(f"[table1] Full grid already complete; skipping ap_pruning", flush=True)
    else:
        t0 = time.time()
        pruning.ap_pruning(
            ports,
            lambda0_list=LAMBDA0_LIST,
            lambda2_list=LAMBDA2_LIST,
            output_dir=out_dir,
            n_train_valid=utils.N_TRAIN_VALID,
            cv_n=utils.CV_N,
            run_full_cv=True,
            kmin=utils.KMIN,
            kmax=utils.KMAX,
            is_tree=True,
            n_workers=args.workers,
        )
        print(f"[table1] Full grid: {time.time()-t0:.1f}s", flush=True)

    factor_mat = pd.read_csv(os.path.join(utils.FACTOR_DIR, "tradable_factors.csv")).iloc[360:636].reset_index(drop=True)

    col_10 = one_column(out_dir, port_path, 10, factor_mat)
    col_40 = one_column(out_dir, port_path, 40, factor_mat)

    table = format_table1(col_10, col_40)
    out_path = os.path.join(utils.OUTPUT_DIR, "tables", "paper_table1_baseline.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    table.to_csv(out_path, index=False)

    print()
    print("=== Paper Table 1 (baseline: median splits, clean CRSP) ===")
    print(table.to_string(index=False))
    print()
    print("Target (paper Table1.png):")
    print("  SDF SR               0.65            0.69")
    print("  α FF3                0.94 [10.11]    0.90 [11.03]")
    print("  α FF5                0.81 [8.76]     0.76 [9.60]")
    print("  α XSF                0.81 [8.77]     0.76 [9.46]")
    print("  α FF11               0.89 [9.12]     0.80 [9.60]")
    print("  XS R²_adj FF3        18.0%           51.0%")
    print("  XS R²_adj FF5        11.0%           64.0%")
    print("  XS R²_adj XSF        28.0%           65.0%")
    print("  XS R²_adj FF11       —               42.0%")
    print()
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
