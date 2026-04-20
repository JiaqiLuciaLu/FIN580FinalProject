"""Python translation of
`reference_code/1_Portfolio_Creation/Tree_Portfolio_Creation/Step4_Filter_SingleSorted_Tree_Ports.R`.

Drop deepest-level single-sorted (monoculture) tree paths.
"""

import os

import pandas as pd

from src.code import utils


# R: filter = c("X1111", "X2222", "X3333"). Python read.csv does not prepend
# "X" to numeric column names, so we match without the leading "X".
FILTER = ["1111", "2222", "3333"]


def filter_tree_ports(feats_list=None, feat1=utils.FEAT1, feat2=utils.FEAT2,
                      tree_portfolio_path=utils.PY_TREE_PORT_DIR):
    if feats_list is None:
        feats_list = utils.FEATS_LIST
    feats_chosen = ["LME", feats_list[feat1 - 1], feats_list[feat2 - 1]]
    sub_dir = "_".join(feats_chosen)

    ports_path = os.path.join(tree_portfolio_path, sub_dir) + "/"
    port_ret = pd.read_csv(os.path.join(ports_path, "level_all_excess_combined.csv"))

    # R: (substring(colnames, 1, 5) %in% filter) & (nchar(colnames) == 11)
    # With the leading-"X" offset removed, this is first-4 chars in FILTER & length 10.
    filt = [(c[:4] in FILTER) and (len(c) == 10) for c in port_ret.columns]
    keep_cols = [c for c, f in zip(port_ret.columns, filt) if not f]
    port_ret = port_ret[keep_cols]

    for f in range(3):
        f_min = pd.read_csv(os.path.join(ports_path, f"level_all_{feats_chosen[f]}_min.csv"))
        f_max = pd.read_csv(os.path.join(ports_path, f"level_all_{feats_chosen[f]}_max.csv"))
        f_min = f_min[keep_cols]
        f_max = f_max[keep_cols]
        f_min.to_csv(
            os.path.join(ports_path, f"level_all_{feats_chosen[f]}_min_filtered.csv"),
            index=False,
        )
        f_max.to_csv(
            os.path.join(ports_path, f"level_all_{feats_chosen[f]}_max_filtered.csv"),
            index=False,
        )

    port_ret.to_csv(
        os.path.join(ports_path, "level_all_excess_combined_filtered.csv"), index=False
    )
