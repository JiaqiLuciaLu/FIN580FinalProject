"""Python translation of `reference_code/main_simplified.R`.

Simplified AP-Trees pipeline: tree-only, single (lambda0, lambda2) point
(no grid search), one K, prints SR + SDF regression results at the end.
Intended as a ~20-minute smoke test of the full pipeline.
"""

import os

import numpy as np

from src.code import utils
from src.code.ap_pruning.ap_pruning import AP_Pruning
from src.code.metrics.pick_best_lambda import pickBestLambda
from src.code.metrics.sdf_timeseries_regressions import SDF_regression
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


def main():
    feats_list = utils.FEATS_LIST
    feat1 = 4   # Operating Prof
    feat2 = 5   # Investment
    y_min = 1964
    y_max = 2016
    n_train_valid = 360  # n months for training and validation
    cvN = 3              # cross-validation counts

    portN = 10

    # Change this to False for max compatibility
    # RunParallel = True
    # pralleln = 6
    RunParallel = False
    pralleln = 1

    raw_data_path = utils.CHAR_PANEL_DIR
    data_chunk_path = utils.PY_DATA_CHUNK_DIR     # clean chunks built below
    tree_portfolio_path = utils.PY_TREE_PORT_DIR
    tree_grid_search_path = utils.PY_TREE_GRID_DIR
    factor_path = utils.FACTOR_DIR
    plot_path = utils.PY_PLOTS_DIR                # noqa: F841

    for p in [data_chunk_path, tree_portfolio_path, tree_grid_search_path]:
        os.makedirs(p, exist_ok=True)

    #####################################################################
    ################  Portfolio Generation from Raw Data ################
    #####################################################################

    # Step 1: build clean yearly chunks from the raw CRSP per-characteristic panels
    # (add_noise=False). R's shipped main.R keeps this commented out because the
    # authors distribute only the noised chunks. We have access to the clean
    # panels under CHAR_PANEL_DIR, so run this step to produce paper-faithful
    # inputs at PY_DATA_CHUNK_DIR/LME_OP_Investment/y{year}.csv.
    create_yearly_chunks(y_min, y_max, feats_list, feat1, feat2,
                         raw_data_path, data_chunk_path, add_noise=False)

    # Steps to generate the tree portfolios
    tree_depth = 4
    create_tree_portfolio(y_min, y_max, tree_depth, feats_list, feat1, feat2,
                          data_chunk_path, tree_portfolio_path, RunParallel, pralleln)
    combinetrees(feats_list, feat1, feat2, tree_depth, factor_path, tree_portfolio_path)
    filter_tree_ports(feats_list, feat1, feat2, tree_portfolio_path)

    #################################################################################
    ################  AP Pruning on Base Portfolios with Grid Search ################
    #################################################################################

    # In the paper we used a much larger grid search on the following parameter,
    # but doing the full parameter search can take a long time to finish
    # lambda0 = np.arange(0, 0.95, 0.05)
    # lambda2 = 0.1 ** np.arange(5, 8.25, 0.25)

    # For demonstration, we put the below parameters not a full grid search
    lambda0 = np.array([0.15])        # R: seq(0.15, 0.15, 0.05)
    lambda2 = np.array([0.1 ** 8])    # R: 0.1^seq(8, 8, 0.25)

    # Due to dimension of trees, the grid search on trees can take a while without
    # parallel computing.
    AP_Pruning(feats_list, feat1, feat2, tree_portfolio_path,
               "level_all_excess_combined_filtered.csv",
               tree_grid_search_path, n_train_valid, cvN, False, 50,
               RunParallel, pralleln, True, lambda0, lambda2)

    #################################################################################
    #############  Parameter Search and Asset Pricing Regression Tests ##############
    #################################################################################

    # Grid Search on Lambda tuning parameters
    sharperatios = pickBestLambda(
        feats_list, feat1, feat2, tree_grid_search_path, portN,
        lambda0, lambda2, tree_portfolio_path,
        "level_all_excess_combined_filtered.csv",
    )

    # Time Series Regression tests wrt factors. The results are collected to form
    # table 1 and 2. Re-run this on the sample where we remove small cap stocks
    # gives table 3.
    regressionresults = SDF_regression(
        feats_list, feat1, feat2, factor_path, tree_grid_search_path,
        "/Selected_Ports_10.csv", "/Selected_Ports_Weights_10.csv",
    )

    print(
        f"Training SR:{round(float(sharperatios[0]), 4)}"
        f"    Validation SR:{round(float(sharperatios[1]), 4)}"
        f"    Testing SR:{round(float(sharperatios[2]), 4)}"
    )
    print("SDF Regression results:")
    print(regressionresults)


if __name__ == "__main__":
    main()
