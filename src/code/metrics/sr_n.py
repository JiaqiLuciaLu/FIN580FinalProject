"""Python translation of `reference_code/3_Metrics_Collection/SR_N.R`.

Loop `pickBestLambda` over K = mink..maxk, cbind the 3-vector results into
a 3xN matrix, write to SR_N.csv.

Optimization (2026-04): the original re-read all 19 x 13 = 247 grid CSVs
once per K (46 K values -> 22,700 reads on /scratch). The grid is now
loaded once via `loadLambdaGrid` and reused across every K — same outputs,
~20x faster on the full paper grid.
"""

import os

import numpy as np
import pandas as pd

from src.code import utils
from src.code.metrics.pick_best_lambda import loadLambdaGrid, pickBestLambda


def pickSRN(feats_list=None, feat1=utils.FEAT1, feat2=utils.FEAT2,
            grid_search_path=utils.PY_TREE_GRID_DIR,
            mink=utils.KMIN, maxk=utils.KMAX,
            lambda0=None, lambda2=None,
            port_path=utils.PY_TREE_PORT_DIR,
            port_file_name="level_all_excess_combined_filtered.csv"):
    if feats_list is None:
        feats_list = utils.FEATS_LIST

    feats = ["LME", feats_list[feat1 - 1], feats_list[feat2 - 1]]
    subdir = "_".join(feats)

    cache = loadLambdaGrid(grid_search_path, subdir, lambda0, lambda2, fullCV=False)

    cols = []
    for k in range(mink, maxk + 1):
        print(k)
        cols.append(
            pickBestLambda(
                feats_list, feat1, feat2, grid_search_path, k,
                lambda0, lambda2, port_path, port_file_name,
                False, True, cache,
            ).reshape(-1, 1)
        )
    srN = np.hstack(cols)

    out = os.path.join(grid_search_path, subdir, "SR_N.csv")
    pd.DataFrame(srN).to_csv(out, index=False)
