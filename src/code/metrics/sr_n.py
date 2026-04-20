"""Python translation of `reference_code/3_Metrics_Collection/SR_N.R`.

Loop `pickBestLambda` over K = mink..maxk, cbind the 3-vector results into
a 3xN matrix, write to SR_N.csv.
"""

import os

import numpy as np
import pandas as pd

from src.code import utils
from src.code.metrics.pick_best_lambda import pickBestLambda


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

    srN = pickBestLambda(
        feats_list, feat1, feat2, grid_search_path, mink,
        lambda0, lambda2, port_path, port_file_name, False,
    )
    srN = srN.reshape(-1, 1)  # (3, 1)

    for k in range(mink + 1, maxk + 1):
        print(k)
        col = pickBestLambda(
            feats_list, feat1, feat2, grid_search_path, k,
            lambda0, lambda2, port_path, port_file_name, False,
        ).reshape(-1, 1)
        srN = np.hstack([srN, col])

    out = os.path.join(grid_search_path, subdir, "SR_N.csv")
    pd.DataFrame(srN).to_csv(out, index=False)
