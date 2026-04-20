"""Python translation of
`reference_code/1_Portfolio_Creation/Tree_Portfolio_Creation/Generate_2Char_Tree_Portfolios_All_Levels_Char_Minmax.R`.

2-characteristic variant of Step 2: feats = (LME, feat1). Trees are built
over 2 features only, but the input chunk files live under the 3-feature
path (LME, feat1, feat2).
"""

import os

import pandas as pd

from src.code import utils
from src.code.portfolio_creation.tree_portfolio_creation.step2_generate_tree_portfolios_all_levels_char_minmax import (
    CNAMES_3,
    CNAMES_4,
    CNAMES_5,
    expand_grid,
)
from src.code.portfolio_creation.tree_portfolio_creation.tree_portfolio_helper import (
    tree_portfolio,
)


def create_tree_portfolio(y_min=utils.Y_MIN, y_max=utils.Y_MAX, tree_depth=4,
                          feats_list=None, feat1=utils.FEAT1, feat2=utils.FEAT2,
                          input_path=utils.DATA_CHUNK_DIR,
                          output_path=utils.PY_TREE_PORT_DIR,
                          runparallel=False, paralleln=1):
    if feats_list is None:
        feats_list = utils.FEATS_LIST
    print(feat1)
    print(feat2)

    if tree_depth == 3:
        cnames = CNAMES_3
    elif tree_depth == 4:
        cnames = CNAMES_4
    elif tree_depth == 5:
        cnames = CNAMES_5
    else:
        raise ValueError(f"tree_depth must be 3, 4, or 5; got {tree_depth}")

    feats = ["LME", feats_list[feat1 - 1]]

    n_feats = len(feats)
    main_dir = output_path
    sub_dir = "_".join(feats)
    os.makedirs(os.path.join(main_dir, sub_dir), exist_ok=True)
    # Chunks live under the 3-feat directory
    data_path = os.path.join(
        input_path,
        "_".join(["LME", feats_list[feat1 - 1], feats_list[feat2 - 1]]),
    ) + "/"

    q_num = 2

    feat_list_base = feats
    feat_list_id_k = expand_grid(n_feats, tree_depth)

    def _run_one(k):
        file_id = "".join(str(x) for x in feat_list_id_k[k])
        feat_list = [feat_list_base[idx - 1] for idx in feat_list_id_k[k]]
        ret = tree_portfolio(data_path, feat_list, tree_depth, q_num,
                             y_min, y_max, "y", feats)
        ret_table = pd.DataFrame(ret[0], columns=[str(c) for c in cnames])
        ret_table.to_csv(
            os.path.join(main_dir, sub_dir, f"{file_id}ret.csv"), index=False
        )

        for f in range(n_feats):
            feat_min_table = pd.DataFrame(ret[2 * f + 1], columns=[str(c) for c in cnames])
            feat_min_table.to_csv(
                os.path.join(main_dir, sub_dir, f"{file_id}{feats[f]}_min.csv"),
                index=False,
            )
            feat_max_table = pd.DataFrame(ret[2 * f + 2], columns=[str(c) for c in cnames])
            feat_max_table.to_csv(
                os.path.join(main_dir, sub_dir, f"{file_id}{feats[f]}_max.csv"),
                index=False,
            )

    if runparallel:
        from multiprocessing import Pool
        with Pool(paralleln) as pool:
            pool.map(_run_one, range(n_feats ** tree_depth))
    else:
        for k in range(n_feats ** tree_depth):
            print(k + 1)
            _run_one(k)
