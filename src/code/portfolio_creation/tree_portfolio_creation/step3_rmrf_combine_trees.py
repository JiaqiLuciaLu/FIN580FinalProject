"""Python translation of
`reference_code/1_Portfolio_Creation/Tree_Portfolio_Creation/Step3_RmRf_Combine_Trees.R`.

Combine portfolios from different trees, dedup higher-level trees, and
subtract the risk-free rate.
"""

import os

import numpy as np
import pandas as pd

from src.code import utils
from src.code.portfolio_creation.tree_portfolio_creation.step2_generate_tree_portfolios_all_levels_char_minmax import (
    expand_grid,
)


def remove_rf(port_ret, factor_path):
    file_nm = os.path.join(factor_path, "rf_factor.csv")
    r_f = pd.read_csv(file_nm, header=None).iloc[:, 0].values.astype(float)
    for i in range(port_ret.shape[1]):
        port_ret.iloc[:, i] = port_ret.iloc[:, i].values - r_f / 100
    return port_ret


def combinetrees(feats_list=None, feat1=utils.FEAT1, feat2=utils.FEAT2,
                 tree_depth=4,
                 factor_path=utils.FACTOR_DIR,
                 tree_sort_path_base=utils.PY_TREE_PORT_DIR):
    if feats_list is None:
        feats_list = utils.FEATS_LIST
    print(feat1)
    print(feat2)
    feats = ["LME", feats_list[feat1 - 1], feats_list[feat2 - 1]]
    n_feats = len(feats)

    tree_sort_path = os.path.join(tree_sort_path_base, "_".join(feats)) + "/"

    feat_list_id_k = expand_grid(n_feats, tree_depth)

    # Build combined ret df
    k = 0
    file_id = "".join(str(x) for x in feat_list_id_k[k])
    file_name = os.path.join(tree_sort_path, f"{file_id}ret.csv")
    port_ret0 = pd.read_csv(file_name)
    port_ret0.columns = [f"{file_id}.{c}" for c in port_ret0.columns]
    port_ret = port_ret0

    for k in range(1, n_feats ** tree_depth):
        file_id = "".join(str(x) for x in feat_list_id_k[k])
        file_name = os.path.join(tree_sort_path, f"{file_id}ret.csv")
        port_ret0 = pd.read_csv(file_name)
        port_ret0.columns = [f"{file_id}.{c}" for c in port_ret0.columns]
        port_ret = pd.concat([port_ret, port_ret0], axis=1)

    # Dedup: keep first occurrence of each unique column (by values)
    arr = port_ret.to_numpy()
    seen = set()
    keep = np.zeros(arr.shape[1], dtype=bool)
    for i in range(arr.shape[1]):
        key = arr[:, i].tobytes()
        if key not in seen:
            seen.add(key)
            keep[i] = True
    port_dedup = port_ret.loc[:, keep]

    port_ret = remove_rf(port_dedup, factor_path)
    print(port_ret.shape[1])
    port_ret.to_csv(
        os.path.join(tree_sort_path, "level_all_excess_combined.csv"), index=False
    )

    for i in range(n_feats):
        # min
        k = 0
        file_id = "".join(str(x) for x in feat_list_id_k[k])
        file_name = os.path.join(tree_sort_path, f"{file_id}{feats[i]}_min.csv")
        port_ret0 = pd.read_csv(file_name)
        port_ret0.columns = [f"{file_id}.{c}" for c in port_ret0.columns]
        port_ret_min = port_ret0

        for k in range(1, n_feats ** tree_depth):
            file_id = "".join(str(x) for x in feat_list_id_k[k])
            file_name = os.path.join(tree_sort_path, f"{file_id}{feats[i]}_min.csv")
            port_ret0 = pd.read_csv(file_name)
            port_ret0.columns = [f"{file_id}.{c}" for c in port_ret0.columns]
            port_ret_min = pd.concat([port_ret_min, port_ret0], axis=1)
        port_ret_min = port_ret_min.loc[:, keep]
        print(port_ret_min.shape[1])
        port_ret_min.to_csv(
            os.path.join(tree_sort_path, f"level_all_{feats[i]}_min.csv"), index=False
        )

        # max
        k = 0
        file_id = "".join(str(x) for x in feat_list_id_k[k])
        file_name = os.path.join(tree_sort_path, f"{file_id}{feats[i]}_max.csv")
        port_ret0 = pd.read_csv(file_name)
        port_ret0.columns = [f"{file_id}.{c}" for c in port_ret0.columns]
        port_ret_max = port_ret0

        for k in range(1, n_feats ** tree_depth):
            file_id = "".join(str(x) for x in feat_list_id_k[k])
            file_name = os.path.join(tree_sort_path, f"{file_id}{feats[i]}_max.csv")
            port_ret0 = pd.read_csv(file_name)
            port_ret0.columns = [f"{file_id}.{c}" for c in port_ret0.columns]
            port_ret_max = pd.concat([port_ret_max, port_ret0], axis=1)
        port_ret_max = port_ret_max.loc[:, keep]
        print(port_ret_max.shape[1])
        port_ret_max.to_csv(
            os.path.join(tree_sort_path, f"level_all_{feats[i]}_max.csv"), index=False
        )
