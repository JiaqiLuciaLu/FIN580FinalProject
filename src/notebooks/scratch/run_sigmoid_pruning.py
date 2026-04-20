"""
LARS-EN grid + pick_best + SDF regression on the S2 sigmoid tree universe.

Mirrors run_figure10.py's flow (988-cell grid, pick_best at K=10 with
full_cv=False matching paper's protocol), but:
  - reads the sigmoid-tree CSV at data/processed/sigmoid_poc/<tag>/<sub>/...
  - writes the grid CSVs to data/processed/sigmoid_poc/<tag>/TreeGridSearch/<sub>/
  - prints the headline comparison (test SDF SR, α, t-stats) vs the paper-style
    baseline so we can directly assess H2 (higher test SDF SR).

Run via slurm/sigmoid_pruning.sbatch. Parameterize the universe with --tag,
e.g. --tag s2_sigmoid_k8.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from src.code import utils
from src.code.portfolio_creation import data_prep
from src.code.ap_pruning import pruning
from src.code.metrics import metrics
from src.code.metrics.regressions import sdf_regression


LAMBDA0_LIST = np.arange(0.0, 0.95, 0.05).tolist()          # 19 values
LAMBDA2_LIST = (10 ** -np.arange(5.0, 8.25, 0.25)).tolist() # 13 values
PORT_N = 10
SUBDIR = utils.SUBDIR_3CHAR  # "LME_OP_Investment"
POC_ROOT = os.path.join(utils.OUTPUT_DIR, "sigmoid_poc")


def run_grid(ports, out_dir, n_workers):
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
        n_workers=n_workers,
    )
    return time.time() - t0


def grid_exists(out_dir):
    for cv in ("cv_1", "cv_2", "cv_3", "full"):
        for i in range(1, len(LAMBDA0_LIST) + 1):
            for j in range(1, len(LAMBDA2_LIST) + 1):
                if not os.path.isfile(os.path.join(out_dir, f"results_{cv}_l0_{i}_l2_{j}.csv")):
                    return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    help="Sigmoid-universe subdir under data/processed/sigmoid_poc, "
                         "e.g. s2_sigmoid_k8")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-grid", action="store_true",
                    help="Reuse existing grid CSVs; run pick_best + sdf regression only.")
    args = ap.parse_args()

    port_path = os.path.join(POC_ROOT, args.tag, SUBDIR,
                             "level_all_excess_combined_filtered.csv")
    if not os.path.isfile(port_path):
        raise SystemExit(f"portfolio file not found: {port_path}")
    out_dir = os.path.join(POC_ROOT, args.tag, "TreeGridSearch", SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[sigmoid_pruning] tag={args.tag}", flush=True)
    print(f"  port_path = {port_path}", flush=True)
    print(f"  out_dir   = {out_dir}", flush=True)

    # load_filtered_tree_portfolios reads <base>/<subdir>/level_all_excess_combined_filtered.csv
    base = os.path.join(POC_ROOT, args.tag)
    ports = data_prep.load_filtered_tree_portfolios(SUBDIR, tree_port_dir=base)
    print(f"[sigmoid_pruning] Ports shape: {ports.shape}", flush=True)

    if args.skip_grid or grid_exists(out_dir):
        print(f"[sigmoid_pruning] Grid already complete; skipping ap_pruning", flush=True)
    else:
        t = run_grid(ports, out_dir, args.workers)
        print(f"[sigmoid_pruning] Full grid: {t:.1f}s", flush=True)

    # pick_best_lambda at K=10, paper protocol (cv_3 only)
    t0 = time.time()
    tr, va, te = metrics.pick_best_lambda(
        result_dir=out_dir,
        portfolio_path=port_path,
        portN=PORT_N,
        lambda0_list=LAMBDA0_LIST,
        lambda2_list=LAMBDA2_LIST,
        full_cv=False,
        write=True,
    )
    print(f"[sigmoid_pruning] pick_best_lambda K={PORT_N}: {time.time()-t0:.1f}s", flush=True)
    print(f"  picked: train_SR={tr:.4f}  valid_SR={va:.4f}  test_SR={te:.4f}", flush=True)

    # Identify which (λ₀, λ₂) got picked
    valid_grid = pd.read_csv(os.path.join(out_dir, f"valid_SR_{PORT_N}.csv"))
    v = valid_grid.to_numpy(float)
    i_star, j_star = np.unravel_index(np.nanargmax(v), v.shape)
    print(f"  picked cell: (λ₀={LAMBDA0_LIST[i_star]:.2f}, λ₂={LAMBDA2_LIST[j_star]:.2e})  "
          f"i*={i_star+1}, j*={j_star+1}", flush=True)

    # pick_sr_n sweeps K in [5,50], produces Fig 10c-style curve
    t0 = time.time()
    metrics.pick_sr_n(
        result_dir=out_dir,
        portfolio_path=port_path,
        lambda0_list=LAMBDA0_LIST,
        lambda2_list=LAMBDA2_LIST,
        kmin=utils.KMIN,
        kmax=utils.KMAX,
        full_cv=False,
    )
    print(f"[sigmoid_pruning] pick_sr_n (K=5..50): {time.time()-t0:.1f}s", flush=True)

    # Read SR_N to print headline Fig 10c-like curve at a few K
    sr_n = pd.read_csv(os.path.join(out_dir, "SR_N.csv"), header=0)
    arr = sr_n.to_numpy(float)  # rows: train/valid/test, cols: K=5..50
    ks = list(range(utils.KMIN, utils.KMAX + 1))
    print(f"  SR by K (test): K=10→{arr[2, ks.index(10)]:.3f}  "
          f"K=20→{arr[2, ks.index(20)]:.3f}  K=40→{arr[2, ks.index(40)]:.3f}", flush=True)

    # SDF regression for α/t at test window
    t0 = time.time()
    df_sdf = sdf_regression(
        result_dir=out_dir,
        factor_path=utils.FACTOR_DIR,
        port_name=f"Selected_Ports_{PORT_N}.csv",
        weight_name=f"Selected_Ports_Weights_{PORT_N}.csv",
    )
    print(f"[sigmoid_pruning] sdf_regression: {time.time()-t0:.1f}s", flush=True)
    print(f"  α [t]:  FF3={df_sdf.iloc[0]['FF3 Alpha']:.3f} [{df_sdf.iloc[0]['FF3 T-Stat']:.2f}]   "
          f"FF5={df_sdf.iloc[0]['FF5 Alpha']:.3f} [{df_sdf.iloc[0]['FF5 T-Stat']:.2f}]   "
          f"XSF={df_sdf.iloc[0]['XSF Alpha']:.3f} [{df_sdf.iloc[0]['XSF T-Stat']:.2f}]   "
          f"FF11={df_sdf.iloc[0]['FF11 Alpha']:.3f} [{df_sdf.iloc[0]['FF11 T-Stat']:.2f}]",
          flush=True)

    print()
    print("=== Comparison with paper-style baseline ===", flush=True)
    print("                                  Paper-style (hard)   Sigmoid S2 k=8", flush=True)
    print(f"  picked (λ₀, λ₂)                (0.90, 1e-8)         ({LAMBDA0_LIST[i_star]:.2f}, {LAMBDA2_LIST[j_star]:.1e})", flush=True)
    print(f"  test SDF SR @ K=10             0.64                  {te:.2f}", flush=True)
    print(f"  α FF3 (R-bug cols)             1.00 [10.33]          {df_sdf.iloc[0]['FF3 Alpha']:.2f} [{df_sdf.iloc[0]['FF3 T-Stat']:.2f}]", flush=True)


if __name__ == "__main__":
    main()
