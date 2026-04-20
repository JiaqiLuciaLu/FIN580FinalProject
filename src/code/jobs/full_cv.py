"""Re-run `pickBestLambda` + `SDF_regression` with fullCV=True.

Derived from `src/code/jobs/main_table1.py`. Differences:
  • Skips portfolio construction AND AP_Pruning — reuses the full 19×13 grid
    already on disk (cv_1 + cv_2 + cv_3 + full results).
  • Passes fullCV=True to every pickBestLambda call. Valid_SR is averaged
    across all 3 CV folds instead of using only cv_3 — motivated by the
    finding that cv_3's regime (1984–1993) systematically inflates SR.
  • Skips pickSRN (pickSRN uses fullCV=False internally and is slow; SR_N is
    not needed for Table 1).
  • Overwrites Selected_Ports_K.csv / Selected_Ports_Weights_K.csv /
    TimeSeriesAlpha.csv under the same processed directories as main_table1.
    Re-run main_table1 afterward to restore fullCV=False artifacts.
"""

import argparse
import os
import time

import numpy as np

from src.code import utils
from src.code.metrics.pick_best_lambda import pickBestLambda
from src.code.metrics.sdf_timeseries_regressions import SDF_regression


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    args = ap.parse_args()

    feats_list = utils.FEATS_LIST
    feat1 = 4
    feat2 = 5

    tree_portfolio_path = utils.PY_TREE_PORT_DIR
    ts32_path = utils.PY_TS_PORT_DIR
    ts64_path = utils.PY_TS64_PORT_DIR
    tree_grid_search_path = utils.PY_TREE_GRID_DIR
    ts32_grid_search_path = utils.PY_TS_GRID_DIR
    ts64_grid_search_path = utils.PY_TS64_GRID_DIR
    factor_path = utils.FACTOR_DIR

    # Same full-paper grid as main_table1.
    lambda0 = np.arange(0, 0.95, 0.05)
    lambda2 = 0.1 ** np.arange(5, 8.25, 0.25)

    subdir = utils.char_subdir(feat1, feat2, feats_list)
    print(f"[full_cv] subdir={subdir}  grid={len(lambda0)}×{len(lambda2)}"
          f"  workers={args.workers} (unused here — pickBestLambda is serial)",
          flush=True)

    # ---- pickBestLambda with fullCV=True on all 5 calls --------------------
    t0 = time.time()
    for K in (10, 40):
        print(f"[full_cv] pickBestLambda tree K={K} fullCV=True", flush=True)
        pickBestLambda(feats_list, feat1, feat2, tree_grid_search_path, K,
                       lambda0, lambda2, tree_portfolio_path,
                       "level_all_excess_combined_filtered.csv",
                       fullCV=True)

    for portN in (10, 32):
        print(f"[full_cv] pickBestLambda ts32 K={portN} fullCV=True", flush=True)
        pickBestLambda(feats_list, feat1, feat2, ts32_grid_search_path, portN,
                       lambda0, lambda2, ts32_path, "excess_ports.csv",
                       fullCV=True)
    for portN in (10, 64):
        print(f"[full_cv] pickBestLambda ts64 K={portN} fullCV=True", flush=True)
        pickBestLambda(feats_list, feat1, feat2, ts64_grid_search_path, portN,
                       lambda0, lambda2, ts64_path, "excess_ports.csv",
                       fullCV=True)
    print(f"[full_cv] pickBestLambda stage: {time.time()-t0:.1f}s", flush=True)

    # ---- SDF regression on the newly picked Selected_Ports -----------------
    t0 = time.time()
    SDF_regression(feats_list, feat1, feat2, factor_path, tree_grid_search_path,
                   "/Selected_Ports_10.csv", "/Selected_Ports_Weights_10.csv")
    SDF_regression(feats_list, feat1, feat2, factor_path, tree_grid_search_path,
                   "/Selected_Ports_40.csv", "/Selected_Ports_Weights_40.csv")
    SDF_regression(feats_list, feat1, feat2, factor_path, ts32_grid_search_path,
                   "/Selected_Ports_32.csv", "/Selected_Ports_Weights_32.csv")
    SDF_regression(feats_list, feat1, feat2, factor_path, ts64_grid_search_path,
                   "/Selected_Ports_64.csv", "/Selected_Ports_Weights_64.csv")
    print(f"[full_cv] SDF_regression stage: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
