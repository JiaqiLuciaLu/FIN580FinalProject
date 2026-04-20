"""Python translation of
`reference_code/1_Portfolio_Creation/Tree_Portfolio_Creation/Combine_2Char_Trees.R`.

Combine min/max feature tables from different trees in the 2-characteristic
(LME, feat1) setting; dedup per feature.
"""

import os

import numpy as np
import pandas as pd

from src.code import utils
from src.code.portfolio_creation.tree_portfolio_creation.step2_generate_tree_portfolios_all_levels_char_minmax import (
    expand_grid,
)


def combinetrees(feats_list=None, feat1=utils.FEAT1, tree_depth=4,
                 factor_path=utils.FACTOR_DIR,
                 tree_sort_path_base=utils.PY_TREE_PORT_DIR):
    if feats_list is None:
        feats_list = utils.FEATS_LIST
    print(feat1)
    feats = ["LME", feats_list[feat1 - 1]]
    n_feats = len(feats)

    tree_sort_path = os.path.join(tree_sort_path_base, "_".join(feats)) + "/"

    feat_list_id_k = expand_grid(n_feats, tree_depth)

    for i in range(n_feats):
        # min
        k = 0
        file_id = "".join(str(x) for x in feat_list_id_k[k])
        file_name = os.path.join(tree_sort_path, f"{file_id}{feats[i]}_min.csv")
        port_ret0 = pd.read_csv(file_name)
        port_ret0.columns = [f"{file_id}.{c}" for c in port_ret0.columns]
        port_ret = port_ret0

        for k in range(1, n_feats ** tree_depth):
            file_id = "".join(str(x) for x in feat_list_id_k[k])
            file_name = os.path.join(tree_sort_path, f"{file_id}{feats[i]}_min.csv")
            port_ret0 = pd.read_csv(file_name)
            port_ret0.columns = [f"{file_id}.{c}" for c in port_ret0.columns]
            port_ret = pd.concat([port_ret, port_ret0], axis=1)

        # Dedup: keep first occurrence of each unique column (by values)
        arr = port_ret.to_numpy()
        seen = set()
        keep = np.zeros(arr.shape[1], dtype=bool)
        for j in range(arr.shape[1]):
            key = arr[:, j].tobytes()
            if key not in seen:
                seen.add(key)
                keep[j] = True
        port_ret = port_ret.loc[:, keep]
        print(port_ret.shape[1])
        port_ret.to_csv(
            os.path.join(tree_sort_path, f"level_all_{feats[i]}_min.csv"), index=False
        )

        # max (uses same `keep` mask from min dedup, per R)
        k = 0
        file_id = "".join(str(x) for x in feat_list_id_k[k])
        file_name = os.path.join(tree_sort_path, f"{file_id}{feats[i]}_max.csv")
        port_ret0 = pd.read_csv(file_name)
        port_ret0.columns = [f"{file_id}.{c}" for c in port_ret0.columns]
        port_ret = port_ret0

        for k in range(1, n_feats ** tree_depth):
            file_id = "".join(str(x) for x in feat_list_id_k[k])
            file_name = os.path.join(tree_sort_path, f"{file_id}{feats[i]}_max.csv")
            port_ret0 = pd.read_csv(file_name)
            port_ret0.columns = [f"{file_id}.{c}" for c in port_ret0.columns]
            port_ret = pd.concat([port_ret, port_ret0], axis=1)
        port_ret = port_ret.loc[:, keep]
        print(port_ret.shape[1])
        port_ret.to_csv(
            os.path.join(tree_sort_path, f"level_all_{feats[i]}_max.csv"), index=False
        )
