"""Benchmark the fuzzy (soft-split) AP-Tree on LME_OP_Investment.

End-to-end pipeline:
  1. Fuzzy step 2  — build ret.csv for all 3^depth permutations
  2. Fuzzy step 3  — concat + dedup + subtract rf
  3. Fuzzy step 4  — drop monoculture depth-4 leaves
  4. AP_Pruning    — LARS-EN grid search
  5. pickBestLambda at K=10
  6. SDF_regression (FF3 / FF5 / XSF / FF11)
  7. SR-vs-K curve (pickSRN)

Per the ablation TODO: start with --grid simple (3×2) to sanity-check the
SR before spending compute on the full 19×13 paper grid. Use --grid full
once the simple pass looks reasonable.

Yearly chunks are assumed to exist at DATA_CHUNK_DIR/LME_OP_Investment/
(reused from the hard-split pipeline — quantile normalization is upstream
of any split rule).

Output goes to PY_FUZZY_TREE_PORT_DIR and PY_FUZZY_TREE_GRID_DIR so the
hard-split outputs under PY_TREE_PORT_DIR / PY_TREE_GRID_DIR stay intact.
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
from src.code.portfolio_creation.tree_portfolio_creation.step2_generate_fuzzy_tree_portfolios_all_levels import (
    DEFAULT_ALPHA,
    create_fuzzy_tree_portfolio,
)
from src.code.portfolio_creation.tree_portfolio_creation.step3_rmrf_combine_fuzzy_trees import (
    combine_fuzzy_trees,
)
from src.code.portfolio_creation.tree_portfolio_creation.step4_filter_singlesorted_fuzzy_tree_ports import (
    filter_fuzzy_tree_ports,
)


def _exists_nonempty(path):
    return os.path.exists(path) and (
        os.path.isfile(path) or len(os.listdir(path)) > 0
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")),
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Sigmoid steepness for the fuzzy split (default: 10.0)",
    )
    ap.add_argument(
        "--dead-threshold",
        type=float,
        default=0.0,
        help="If > 0, zero out per-firm leaf weights below this cutoff before "
             "computing leaf returns (renormalizes the survivors). Used to test "
             "whether the soft-tail of dead stocks is overfitting noise. "
             "Default 0 = no pruning (standard fuzzy build).",
    )
    ap.add_argument("--feat1", type=int, default=4)   # OP
    ap.add_argument("--feat2", type=int, default=5)   # Investment
    ap.add_argument("--tree-depth", type=int, default=4)
    ap.add_argument(
        "--grid",
        choices=["simple", "full"],
        default="simple",
        help="simple: 3x2 pilot grid; full: paper 19x13 grid.",
    )
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of fuzzy step 2/3/4 outputs.",
    )
    args = ap.parse_args()

    feats_list = utils.FEATS_LIST
    y_min, y_max = 1964, 2016
    n_train_valid = 360
    cvN = 3
    kmax = 50
    portN = 10
    run_parallel = True
    workers = max(1, args.workers)

    subdir = utils.char_subdir(args.feat1, args.feat2, feats_list)
    data_chunk_path = utils.PY_DATA_CHUNK_DIR
    fuzzy_port_path = utils.PY_FUZZY_TREE_PORT_DIR
    fuzzy_grid_path = utils.PY_FUZZY_TREE_GRID_DIR
    factor_path = utils.FACTOR_DIR

    os.makedirs(fuzzy_port_path, exist_ok=True)
    os.makedirs(fuzzy_grid_path, exist_ok=True)

    print(
        f"[fuzzy_bench] subdir={subdir} alpha={args.alpha} depth={args.tree_depth} "
        f"grid={args.grid} workers={workers} rebuild={args.rebuild} "
        f"dead_threshold={args.dead_threshold}",
        flush=True,
    )

    # --------- step 2: fuzzy tree portfolios ----------------------------------
    t0 = time.time()
    step2_marker = os.path.join(
        fuzzy_port_path, subdir, f"{'1' * args.tree_depth}ret.csv"
    )
    if args.rebuild or not _exists_nonempty(step2_marker):
        print("[fuzzy_bench] step 2: fuzzy tree portfolios", flush=True)
        create_fuzzy_tree_portfolio(
            y_min=y_min,
            y_max=y_max,
            tree_depth=args.tree_depth,
            feats_list=feats_list,
            feat1=args.feat1,
            feat2=args.feat2,
            input_path=data_chunk_path,
            output_path=fuzzy_port_path,
            alpha=args.alpha,
            dead_threshold=args.dead_threshold,
            runparallel=run_parallel,
            paralleln=workers,
        )
    else:
        print("[fuzzy_bench] step 2 skipped (cached)", flush=True)
    print(f"[fuzzy_bench] step 2: {time.time() - t0:.1f}s", flush=True)

    # --------- step 3: combine + rmrf -----------------------------------------
    t0 = time.time()
    step3_marker = os.path.join(fuzzy_port_path, subdir, "level_all_excess_combined.csv")
    if args.rebuild or not _exists_nonempty(step3_marker):
        print("[fuzzy_bench] step 3: combine + rmrf", flush=True)
        combine_fuzzy_trees(
            feats_list=feats_list,
            feat1=args.feat1,
            feat2=args.feat2,
            tree_depth=args.tree_depth,
            factor_path=factor_path,
            tree_sort_path_base=fuzzy_port_path,
        )
    else:
        print("[fuzzy_bench] step 3 skipped (cached)", flush=True)
    print(f"[fuzzy_bench] step 3: {time.time() - t0:.1f}s", flush=True)

    # --------- step 4: monoculture filter -------------------------------------
    t0 = time.time()
    step4_marker = os.path.join(
        fuzzy_port_path, subdir, "level_all_excess_combined_filtered.csv"
    )
    if args.rebuild or not _exists_nonempty(step4_marker):
        print("[fuzzy_bench] step 4: monoculture filter", flush=True)
        filter_fuzzy_tree_ports(
            feats_list=feats_list,
            feat1=args.feat1,
            feat2=args.feat2,
            tree_portfolio_path=fuzzy_port_path,
        )
    else:
        print("[fuzzy_bench] step 4 skipped (cached)", flush=True)
    print(f"[fuzzy_bench] step 4: {time.time() - t0:.1f}s", flush=True)

    # --------- grid choice -----------------------------------------------------
    if args.grid == "full":
        lambda0 = np.arange(0, 0.95, 0.05)
        lambda2 = 0.1 ** np.arange(5, 8.25, 0.25)
    else:
        lambda0 = np.array([0.0, 0.15, 0.30])
        lambda2 = np.array([1e-5, 1e-7])
    print(
        f"[fuzzy_bench] lambda grid: {len(lambda0)} x {len(lambda2)} = "
        f"{len(lambda0) * len(lambda2)} cells",
        flush=True,
    )

    # --------- AP_Pruning ------------------------------------------------------
    t0 = time.time()
    AP_Pruning(
        feats_list, args.feat1, args.feat2,
        fuzzy_port_path,
        "level_all_excess_combined_filtered.csv",
        fuzzy_grid_path,
        n_train_valid, cvN,
        True,     # runFullCV
        kmax,
        run_parallel, workers,
        True,     # IsTree
        lambda0, lambda2,
    )
    print(f"[fuzzy_bench] AP_Pruning: {time.time() - t0:.1f}s", flush=True)

    # --------- pickBestLambda + SDF + SR curve --------------------------------
    pickBestLambda(
        feats_list, args.feat1, args.feat2,
        fuzzy_grid_path, portN,
        lambda0, lambda2,
        fuzzy_port_path,
        "level_all_excess_combined_filtered.csv",
    )

    SDF_regression(
        feats_list, args.feat1, args.feat2, factor_path,
        fuzzy_grid_path,
        f"/Selected_Ports_{portN}.csv",
        f"/Selected_Ports_Weights_{portN}.csv",
    )

    pickSRN(
        feats_list, args.feat1, args.feat2,
        fuzzy_grid_path, 5, kmax,
        lambda0, lambda2,
        fuzzy_port_path,
        "level_all_excess_combined_filtered.csv",
    )

    print("[fuzzy_bench] done", flush=True)


if __name__ == "__main__":
    main()
