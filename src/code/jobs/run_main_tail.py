"""Re-run only the tail-end pieces of main.py that crashed last time:
  • pickSRN (SR vs K curve, K=5..50) → SR_N.csv — feeds Figure 10
  • create_2char_tree_portfolio + combinetrees_2char → Figure 1b inputs

Relies on outputs from the prior main.py run (tree/triple-sort/AP_Pruning
already on disk). No need to rebuild Step 1-9.
"""

import os

import numpy as np

from src.code import utils
from src.code.metrics.sr_n import pickSRN
from src.code.portfolio_creation.tree_portfolio_creation.combine_2char_trees import (
    combinetrees as combinetrees_2char,
)
from src.code.portfolio_creation.tree_portfolio_creation.generate_2char_tree_portfolios_all_levels_char_minmax import (
    create_tree_portfolio as create_2char_tree_portfolio,
)


def main():
    feats_list = utils.FEATS_LIST
    feat1 = 4   # Operating Prof
    feat2 = 5   # Investment
    y_min = 1964
    y_max = 2016
    tree_depth = 4

    data_chunk_path = utils.PY_DATA_CHUNK_DIR          # clean chunks from prior run
    tree_portfolio_path = utils.PY_TREE_PORT_DIR
    tree_grid_search_path = utils.PY_TREE_GRID_DIR
    factor_path = utils.FACTOR_DIR

    lambda0 = np.linspace(0.5, 0.6, 3)          # R: seq(0.5, 0.6, 0.05)
    lambda2 = 0.1 ** np.linspace(7, 7.5, 3)     # R: 0.1^seq(7, 7.5, 0.25)

    # Step A: SR_N curve for tree (K=5..50) — Figure 10 input.
    # With the all-NaN guard in pickBestLambda, any K where no grid cell has a
    # matching portsN row will produce a NaN column in SR_N.csv instead of crashing.
    print("=== pickSRN on tree (K=5..50) ===")
    pickSRN(feats_list, feat1, feat2, tree_grid_search_path, 5, 50,
            lambda0, lambda2, tree_portfolio_path,
            "level_all_excess_combined_filtered.csv")

    # Step B: 2-char tree + combine_2char for Figure 1b bounding-box plot.
    print("=== 2-char tree: create + combine (feats=LME, OP) ===")
    create_2char_tree_portfolio(y_min, y_max, tree_depth, feats_list, feat1, feat2,
                                data_chunk_path, tree_portfolio_path, False, 0)
    combinetrees_2char(feats_list, feat1, tree_depth, factor_path, tree_portfolio_path)

    print("=== run_main_tail complete ===")


if __name__ == "__main__":
    main()
