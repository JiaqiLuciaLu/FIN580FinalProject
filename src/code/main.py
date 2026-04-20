"""Python translation of `reference_code/main.R`.

Full AP-Trees pipeline driver: tree + triple-sort(32,64) portfolio creation,
AP pruning grid search on all three, pick-best-lambda, SDF alpha regressions,
SR-vs-K curve, and the 2-char tree outputs used by Figure 1.

Tested dependency versions (Python side):
  numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib
"""

import os

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
    data_chunk_path = utils.PY_DATA_CHUNK_DIR           # clean chunks built below
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
    OrderPath = os.path.join(utils.DATA_DIR, "Summary", "SR.csv")  # noqa: F841

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

    # Steps to generate the triple sort 32 portfolios
    gen_triple_sort_32(feats_list, feat1, feat2, y_min, y_max,
                       data_chunk_path, ts32_path, factor_path)

    # Steps to generate the triple sort 64 portfolios
    gen_triple_sort_64(feats_list, feat1, feat2, y_min, y_max,
                       data_chunk_path, ts64_path, factor_path)

    #################################################################################
    ################  AP Pruning on Base Portfolios with Grid Search ################
    #################################################################################

    # In the paper we used a much larger grid search on the following parameter,
    # but doing the full parameter search can take a long time to finish
    # lambda0 = np.arange(0, 0.95, 0.05)
    # lambda2 = 0.1 ** np.arange(5, 8.25, 0.25)

    # For demonstration, we put the below smaller grid of hyper-parameters
    lambda0 = np.linspace(0.5, 0.6, 3)          # R: seq(0.5, 0.6, 0.05)
    lambda2 = 0.1 ** np.linspace(7, 7.5, 3)     # R: 0.1^seq(7, 7.5, 0.25)

    # Due to dimension of trees, the grid search on trees can take a while without
    # parallel computing.
    AP_Pruning(feats_list, feat1, feat2, tree_portfolio_path,
               "level_all_excess_combined_filtered.csv",
               tree_grid_search_path, n_train_valid, cvN, False, 50,
               RunParallel, pralleln, True, lambda0, lambda2)
    AP_Pruning(feats_list, feat1, feat2, ts32_path, "excess_ports.csv",
               ts32_grid_search_path, n_train_valid, cvN, True, 32,
               RunParallel, pralleln, False, lambda0, lambda2)
    AP_Pruning(feats_list, feat1, feat2, ts64_path, "excess_ports.csv",
               ts64_grid_search_path, n_train_valid, cvN, True, 64,
               RunParallel, pralleln, False, lambda0, lambda2)

    #################################################################################
    #############  Parameter Search and Asset Pricing Regression Tests ##############
    #################################################################################

    # Grid Search on Lambda tuning parameters
    pickBestLambda(feats_list, feat1, feat2, tree_grid_search_path, portN,
                   lambda0, lambda2, tree_portfolio_path,
                   "level_all_excess_combined_filtered.csv")
    pickBestLambda(feats_list, feat1, feat2, ts32_grid_search_path, portN,
                   lambda0, lambda2, ts32_path, "excess_ports.csv")
    pickBestLambda(feats_list, feat1, feat2, ts64_grid_search_path, portN,
                   lambda0, lambda2, ts64_path, "excess_ports.csv")
    pickBestLambda(feats_list, feat1, feat2, ts32_grid_search_path, 32,
                   lambda0, lambda2, ts32_path, "excess_ports.csv")
    pickBestLambda(feats_list, feat1, feat2, ts64_grid_search_path, 64,
                   lambda0, lambda2, ts64_path, "excess_ports.csv")

    # Time Series Regression tests wrt factors. The results are collected to form
    # table 1 and 2. Re-run this on the sample where we remove small cap stocks
    # gives table 3.
    SDF_regression(feats_list, feat1, feat2, factor_path, tree_grid_search_path,
                   "/Selected_Ports_10.csv", "/Selected_Ports_Weights_10.csv")
    SDF_regression(feats_list, feat1, feat2, factor_path, ts32_grid_search_path,
                   "/Selected_Ports_10.csv", "/Selected_Ports_Weights_10.csv")
    SDF_regression(feats_list, feat1, feat2, factor_path, ts64_grid_search_path,
                   "/Selected_Ports_10.csv", "/Selected_Ports_Weights_10.csv")

    # Collect Sharpe Ratio results by different number of portfolios
    pickSRN(feats_list, feat1, feat2, tree_grid_search_path, 5, 50,
            lambda0, lambda2, tree_portfolio_path,
            "level_all_excess_combined_filtered.csv")

    #################################################################################
    ####################################  Plots  ####################################
    #################################################################################

    ### Used to generate the base empirical bounds for Figure 1; actual code to
    ### make the plots is in Python (`src/code/plots/figure1bc_empirical_portfolio_bounding_box.py`).
    create_2char_tree_portfolio(y_min, y_max, tree_depth, feats_list, feat1, feat2,
                                data_chunk_path, tree_portfolio_path, False, 0)
    combinetrees_2char(feats_list, feat1, tree_depth, factor_path, tree_portfolio_path)

    # The remaining figure scripts from R (Figure6a_7_8, Figure6b_C2ab_C4_C5,
    # Figure6c_C2c, Figure9a, Figure9b, Figure10ac, Figure13_C6ab, Figure14,
    # Figure15, FigureC8ab) have not yet been ported to Python.


if __name__ == "__main__":
    main()
