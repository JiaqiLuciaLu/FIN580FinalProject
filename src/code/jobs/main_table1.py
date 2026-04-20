"""Full-grid baseline run for paper Table 1 (cross-section LME_OP_Investment).

Derived from `src/code/main.py`. Changes vs main.py:
  1. λ₀, λ₂ grids expanded from the 3×3 demo to the paper's full 19×13.
  2. RunParallel=True; worker count via --workers (default = $SLURM_CPUS_PER_TASK
     or 8).
  3. Portfolio-construction steps (yearly chunks, tree, TS32, TS64, 2-char tree)
     are skipped if their outputs already exist on disk. Use --rebuild to force.
  4. pickBestLambda and SDF_regression extended to K=40 for the tree.
  5. TS32 / TS64 SDF_regression uses Selected_Ports_{32,64}.csv (main.py passed
     Selected_Ports_10.csv, which does not match Table 1).

Does NOT include XS R²_adj — that's a small local computation, run separately.
"""

import argparse
import os
import time

import numpy as np

from src.code import utils
from src.code.ap_pruning.ap_pruning import AP_Pruning
from src.code.metrics.pick_best_lambda import pickBestLambda
from src.code.metrics.sdf_timeseries_regressions import SDF_regression
from src.code.metrics.sr_n import pickSRN
from src.code.portfolio_creation.tree_portfolio_creation.combine_2char_trees import (
    combinetrees as combinetrees_2char,
)
from src.code.portfolio_creation.tree_portfolio_creation.generate_2char_tree_portfolios_all_levels_char_minmax import (
    create_tree_portfolio as create_2char_tree_portfolio,
)
from src.code.portfolio_creation.tree_portfolio_creation.step1_combine_raw_chars_convert_quantile_split_yearly_chunks import (
    create_yearly_chunks,
)
from src.code.portfolio_creation.tree_portfolio_creation.step2_generate_tree_portfolios_all_levels_char_minmax import (
    create_tree_portfolio,
)
from src.code.portfolio_creation.tree_portfolio_creation.step3_rmrf_combine_trees import (
    combinetrees,
)
from src.code.portfolio_creation.tree_portfolio_creation.step4_filter_singlesorted_tree_ports import (
    filter_tree_ports,
)
from src.code.portfolio_creation.triple_sort_portfolio_creation.triple_sort_32_portfolios import (
    gen_triple_sort_32,
)
from src.code.portfolio_creation.triple_sort_portfolio_creation.triple_sort_64_portfolios import (
    gen_triple_sort_64,
)


def _exists(path):
    return os.path.exists(path) and (os.path.isfile(path) or len(os.listdir(path)) > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
                    help="Parallel workers for AP_Pruning (default = SLURM_CPUS_PER_TASK or 8).")
    ap.add_argument("--rebuild", action="store_true",
                    help="Force rebuild of portfolio construction steps (default: skip if cached).")
    args = ap.parse_args()

    feats_list = utils.FEATS_LIST
    feat1 = 4   # Operating Prof
    feat2 = 5   # Investment
    y_min = 1964
    y_max = 2016
    n_train_valid = 360
    cvN = 3

    portN = 10

    # --- Change 2: parallel on ------------------------------------------------
    RunParallel = True
    pralleln = max(1, args.workers)

    raw_data_path = utils.CHAR_PANEL_DIR
    data_chunk_path = utils.PY_DATA_CHUNK_DIR
    tree_portfolio_path = utils.PY_TREE_PORT_DIR
    ts32_path = utils.PY_TS_PORT_DIR
    ts64_path = utils.PY_TS64_PORT_DIR
    tree_grid_search_path = utils.PY_TREE_GRID_DIR
    ts32_grid_search_path = utils.PY_TS_GRID_DIR
    ts64_grid_search_path = utils.PY_TS64_GRID_DIR
    factor_path = utils.FACTOR_DIR
    plot_path = utils.PY_PLOTS_DIR

    for p in [data_chunk_path, tree_portfolio_path, ts32_path, ts64_path,
              tree_grid_search_path, ts32_grid_search_path, ts64_grid_search_path,
              plot_path]:
        os.makedirs(p, exist_ok=True)

    subdir = utils.char_subdir(feat1, feat2, feats_list)  # LME_OP_Investment

    print(f"[main_table1] workers={pralleln}, rebuild={args.rebuild}, subdir={subdir}",
          flush=True)

    # =========================================================================
    # Change 3: Portfolio construction — skip if outputs exist
    # =========================================================================
    t0 = time.time()

    chunks_marker = os.path.join(data_chunk_path, subdir, f"y{y_max}.csv")
    if args.rebuild or not _exists(chunks_marker):
        print("[main_table1] Step 1: yearly chunks", flush=True)
        create_yearly_chunks(y_min, y_max, feats_list, feat1, feat2,
                             raw_data_path, data_chunk_path, add_noise=False)
    else:
        print("[main_table1] Step 1 skipped (chunks cached)", flush=True)

    tree_marker = os.path.join(tree_portfolio_path, subdir,
                               "level_all_excess_combined_filtered.csv")
    if args.rebuild or not _exists(tree_marker):
        print("[main_table1] Steps 2-4: tree portfolios", flush=True)
        tree_depth = 4
        create_tree_portfolio(y_min, y_max, tree_depth, feats_list, feat1, feat2,
                              data_chunk_path, tree_portfolio_path, RunParallel, pralleln)
        combinetrees(feats_list, feat1, feat2, tree_depth, factor_path, tree_portfolio_path)
        filter_tree_ports(feats_list, feat1, feat2, tree_portfolio_path)
    else:
        print("[main_table1] Steps 2-4 skipped (tree cached)", flush=True)

    ts32_marker = os.path.join(ts32_path, subdir, "excess_ports.csv")
    if args.rebuild or not _exists(ts32_marker):
        print("[main_table1] Triple-sort 32", flush=True)
        gen_triple_sort_32(feats_list, feat1, feat2, y_min, y_max,
                           data_chunk_path, ts32_path, factor_path)
    else:
        print("[main_table1] TS32 skipped (cached)", flush=True)

    ts64_marker = os.path.join(ts64_path, subdir, "excess_ports.csv")
    if args.rebuild or not _exists(ts64_marker):
        print("[main_table1] Triple-sort 64", flush=True)
        gen_triple_sort_64(feats_list, feat1, feat2, y_min, y_max,
                           data_chunk_path, ts64_path, factor_path)
    else:
        print("[main_table1] TS64 skipped (cached)", flush=True)

    print(f"[main_table1] Portfolio stage: {time.time()-t0:.1f}s", flush=True)

    # =========================================================================
    # Change 1: Full paper grid (19 × 13)
    # =========================================================================
    lambda0 = np.arange(0, 0.95, 0.05)
    lambda2 = 0.1 ** np.arange(5, 8.25, 0.25)
    print(f"[main_table1] lambda grid: {len(lambda0)} × {len(lambda2)} = "
          f"{len(lambda0)*len(lambda2)} cells / CV fold", flush=True)

    # =========================================================================
    # AP-Pruning with runFullCV=True (required for pickBestLambda)
    # =========================================================================
    t0 = time.time()
    AP_Pruning(feats_list, feat1, feat2, tree_portfolio_path,
               "level_all_excess_combined_filtered.csv",
               tree_grid_search_path, n_train_valid, cvN,
               True,   # runFullCV
               50,     # kmax
               RunParallel, pralleln,
               True,   # IsTree
               lambda0, lambda2)
    print(f"[main_table1] AP_Pruning tree: {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    AP_Pruning(feats_list, feat1, feat2, ts32_path, "excess_ports.csv",
               ts32_grid_search_path, n_train_valid, cvN,
               True, 32, RunParallel, pralleln,
               False,  # IsTree
               lambda0, lambda2)
    print(f"[main_table1] AP_Pruning ts32: {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    AP_Pruning(feats_list, feat1, feat2, ts64_path, "excess_ports.csv",
               ts64_grid_search_path, n_train_valid, cvN,
               True, 64, RunParallel, pralleln,
               False,
               lambda0, lambda2)
    print(f"[main_table1] AP_Pruning ts64: {time.time()-t0:.1f}s", flush=True)

    # =========================================================================
    # Change 4: pickBestLambda — tree at K=10 AND K=40
    # =========================================================================
    for K in (10, 40):
        pickBestLambda(feats_list, feat1, feat2, tree_grid_search_path, K,
                       lambda0, lambda2, tree_portfolio_path,
                       "level_all_excess_combined_filtered.csv")

    pickBestLambda(feats_list, feat1, feat2, ts32_grid_search_path, portN,
                   lambda0, lambda2, ts32_path, "excess_ports.csv")
    pickBestLambda(feats_list, feat1, feat2, ts32_grid_search_path, 32,
                   lambda0, lambda2, ts32_path, "excess_ports.csv")
    pickBestLambda(feats_list, feat1, feat2, ts64_grid_search_path, portN,
                   lambda0, lambda2, ts64_path, "excess_ports.csv")
    pickBestLambda(feats_list, feat1, feat2, ts64_grid_search_path, 64,
                   lambda0, lambda2, ts64_path, "excess_ports.csv")

    # =========================================================================
    # Change 4 + 5: SDF regressions
    #   Tree: K=10 and K=40
    #   TS32: K=32   (was wrongly K=10 in main.py)
    #   TS64: K=64   (was wrongly K=10 in main.py)
    # =========================================================================
    SDF_regression(feats_list, feat1, feat2, factor_path, tree_grid_search_path,
                   "/Selected_Ports_10.csv", "/Selected_Ports_Weights_10.csv")
    SDF_regression(feats_list, feat1, feat2, factor_path, tree_grid_search_path,
                   "/Selected_Ports_40.csv", "/Selected_Ports_Weights_40.csv")
    SDF_regression(feats_list, feat1, feat2, factor_path, ts32_grid_search_path,
                   "/Selected_Ports_32.csv", "/Selected_Ports_Weights_32.csv")
    SDF_regression(feats_list, feat1, feat2, factor_path, ts64_grid_search_path,
                   "/Selected_Ports_64.csv", "/Selected_Ports_Weights_64.csv")

    # SR-vs-K curve (for Figure 10a/c — already unblocks on the old grid, but
    # re-computing on the full grid gives the paper-consistent values).
    pickSRN(feats_list, feat1, feat2, tree_grid_search_path, 5, 50,
            lambda0, lambda2, tree_portfolio_path,
            "level_all_excess_combined_filtered.csv")

    # 2-char tree for Figure 1b — skip if cached.
    two_char_marker = os.path.join(
        tree_portfolio_path,
        "_".join(["LME", feats_list[feat1 - 1]]),
        "level_all_LME_min.csv",
    )
    if args.rebuild or not _exists(two_char_marker):
        tree_depth = 4
        create_2char_tree_portfolio(y_min, y_max, tree_depth, feats_list,
                                    feat1, feat2, data_chunk_path,
                                    tree_portfolio_path, False, 0)
        combinetrees_2char(feats_list, feat1, tree_depth, factor_path,
                           tree_portfolio_path)
    else:
        print("[main_table1] 2-char tree skipped (cached)", flush=True)


if __name__ == "__main__":
    main()
