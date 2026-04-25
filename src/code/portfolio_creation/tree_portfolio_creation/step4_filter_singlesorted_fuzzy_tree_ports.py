"""Fuzzy analogue of `step4_filter_singlesorted_tree_ports.py`: ret only.

The monoculture filter is name-based — it drops depth-4 leaves from
permutations that split on a single characteristic at every level
(file_ids "1111", "2222", "3333", each with 16 depth-4 leaves). That
semantics is unchanged for fuzzy splits: a same-char 4-split tree is
still a fine-grained single-sort regardless of soft / hard split.
"""

import os

import pandas as pd

from src.code import utils


# See step4_filter_singlesorted_tree_ports.py for the format rationale.
FILTER = ["1111", "2222", "3333"]


def filter_fuzzy_tree_ports(
    feats_list=None,
    feat1=utils.FEAT1,
    feat2=utils.FEAT2,
    tree_portfolio_path=utils.PY_FUZZY_TREE_PORT_DIR,
):
    if feats_list is None:
        feats_list = utils.FEATS_LIST
    feats_chosen = ["LME", feats_list[feat1 - 1], feats_list[feat2 - 1]]
    sub_dir = "_".join(feats_chosen)
    ports_path = os.path.join(tree_portfolio_path, sub_dir) + "/"

    port_ret = pd.read_csv(os.path.join(ports_path, "level_all_excess_combined.csv"))

    filt = [(c[:4] in FILTER) and (len(c) == 10) for c in port_ret.columns]
    keep_cols = [c for c, f in zip(port_ret.columns, filt) if not f]
    port_ret = port_ret[keep_cols]

    port_ret.to_csv(
        os.path.join(ports_path, "level_all_excess_combined_filtered.csv"),
        index=False,
    )
    print(f"[fuzzy step4] kept {len(keep_cols)} columns after monoculture filter")
