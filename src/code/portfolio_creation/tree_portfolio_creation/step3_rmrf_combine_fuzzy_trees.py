"""Fuzzy analogue of `step3_rmrf_combine_trees.py`: ret only.

Concatenates `{file_id}ret.csv` across the 3**tree_depth permutations,
dedups columns with identical values, subtracts the risk-free rate, and
writes `level_all_excess_combined.csv`. Skips the `{feat}_min`/`_max`
files (fuzzy step 2 does not emit them — every firm is in every leaf,
so per-leaf min/max are meaningless; flagged in `docs/abalation_1.md`).
"""

import os

import numpy as np
import pandas as pd

from src.code import utils
from src.code.portfolio_creation.tree_portfolio_creation.step2_generate_tree_portfolios_all_levels_char_minmax import (
    expand_grid,
)


def _remove_rf(port_ret, factor_path):
    r_f = (
        pd.read_csv(os.path.join(factor_path, "rf_factor.csv"), header=None)
        .iloc[:, 0]
        .values.astype(float)
    )
    for i in range(port_ret.shape[1]):
        port_ret.iloc[:, i] = port_ret.iloc[:, i].values - r_f / 100
    return port_ret


def combine_fuzzy_trees(
    feats_list=None,
    feat1=utils.FEAT1,
    feat2=utils.FEAT2,
    tree_depth=4,
    factor_path=utils.FACTOR_DIR,
    tree_sort_path_base=utils.PY_FUZZY_TREE_PORT_DIR,
):
    if feats_list is None:
        feats_list = utils.FEATS_LIST
    feats = ["LME", feats_list[feat1 - 1], feats_list[feat2 - 1]]
    n_feats = len(feats)

    tree_sort_path = os.path.join(tree_sort_path_base, "_".join(feats)) + "/"
    feat_list_id_k = expand_grid(n_feats, tree_depth)

    parts = []
    for k in range(n_feats ** tree_depth):
        file_id = "".join(str(x) for x in feat_list_id_k[k])
        df = pd.read_csv(os.path.join(tree_sort_path, f"{file_id}ret.csv"))
        df.columns = [f"{file_id}.{c}" for c in df.columns]
        parts.append(df)
    port_ret = pd.concat(parts, axis=1)

    # Dedup: keep first occurrence of each unique column by bit-identical values.
    arr = port_ret.to_numpy()
    seen = set()
    keep = np.zeros(arr.shape[1], dtype=bool)
    for i in range(arr.shape[1]):
        key = arr[:, i].tobytes()
        if key not in seen:
            seen.add(key)
            keep[i] = True
    port_ret = port_ret.loc[:, keep]

    port_ret = _remove_rf(port_ret, factor_path)
    print(f"[fuzzy step3] combined columns: {port_ret.shape[1]}")
    port_ret.to_csv(
        os.path.join(tree_sort_path, "level_all_excess_combined.csv"), index=False
    )
