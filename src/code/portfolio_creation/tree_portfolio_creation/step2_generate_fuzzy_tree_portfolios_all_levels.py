"""Fuzzy (soft-split) analogue of
`step2_generate_tree_portfolios_all_levels_char_minmax.py`.

Builds AP-Tree portfolios via the per-firm sigmoid split (see
`fuzzy_tree_split.fuzzy_split` and `docs/abalation_1.md`). For each
characteristic permutation it writes `{file_id}ret.csv` with columns
named by the same breadth-first node codes the hard-split pipeline
uses (1, 11, 12, 111, ..., 122, 1111, ...), so `step3_rmrf_combine_trees.py`
and every downstream stage can read the fuzzy output without changes.

v1 scope: ret-only (no `{file_id}{feat}_min.csv` / `_max.csv` — every
firm is in every leaf, so per-leaf min/max collapse to cross-section
bounds). If the monoculture filter in step 4 requires those, add them
as weighted percentiles later.

Output dir: `utils.PY_FUZZY_TREE_PORT_DIR / <LME_feat1_feat2> /` so
the hard-split outputs under `PY_TREE_PORT_DIR` stay untouched.
"""

import os

import pandas as pd

from src.code import utils
from src.code.portfolio_creation.tree_portfolio_creation.fuzzy_tree_portfolio_helper import (
    fuzzy_tree_portfolio,
)
from src.code.portfolio_creation.tree_portfolio_creation.step2_generate_tree_portfolios_all_levels_char_minmax import (
    CNAMES_3,
    CNAMES_4,
    CNAMES_5,
    expand_grid,
)


DEFAULT_ALPHA = 10.0


def _build_one_tree_permutation(
    k,
    feat_list_id_k,
    feats,
    data_path,
    tree_depth,
    y_min,
    y_max,
    cnames,
    main_dir,
    sub_dir,
    alpha,
    dead_threshold,
):
    """Module-level worker for `multiprocessing.Pool` (must be picklable)."""
    file_id = "".join(str(x) for x in feat_list_id_k[k])
    feat_list = [feats[idx - 1] for idx in feat_list_id_k[k]]
    ret_table = fuzzy_tree_portfolio(
        data_path=data_path,
        feat_list=feat_list,
        tree_depth=tree_depth,
        y_min=y_min,
        y_max=y_max,
        file_prefix="y",
        alpha=alpha,
        dead_threshold=dead_threshold,
    )
    df = pd.DataFrame(ret_table, columns=[str(c) for c in cnames])
    df.to_csv(os.path.join(main_dir, sub_dir, f"{file_id}ret.csv"), index=False)


def create_fuzzy_tree_portfolio(
    y_min=utils.Y_MIN,
    y_max=utils.Y_MAX,
    tree_depth=4,
    feats_list=None,
    feat1=utils.FEAT1,
    feat2=utils.FEAT2,
    input_path=utils.DATA_CHUNK_DIR,
    output_path=utils.PY_FUZZY_TREE_PORT_DIR,
    alpha=DEFAULT_ALPHA,
    dead_threshold=0.0,
    runparallel=False,
    paralleln=1,
):
    """Build fuzzy tree portfolios for every permutation of the 3 chars
    (LME + feat1 + feat2) at the given tree depth and sigmoid steepness.

    Mirrors `create_tree_portfolio(...)` one-for-one except:
        - calls `fuzzy_tree_portfolio` instead of `tree_portfolio`;
        - no `q_num` (binary only);
        - no `feat_min`/`feat_max` CSVs (v1);
        - takes `alpha`.
    """
    if feats_list is None:
        feats_list = utils.FEATS_LIST
    print(f"feat1={feat1}, feat2={feat2}, alpha={alpha}, dead_threshold={dead_threshold}")

    if tree_depth == 3:
        cnames = CNAMES_3
    elif tree_depth == 4:
        cnames = CNAMES_4
    elif tree_depth == 5:
        cnames = CNAMES_5
    else:
        raise ValueError(f"tree_depth must be 3, 4, or 5; got {tree_depth}")

    feats = ["LME", feats_list[feat1 - 1], feats_list[feat2 - 1]]
    n_feats = len(feats)
    main_dir = output_path
    sub_dir = "_".join(feats)
    os.makedirs(os.path.join(main_dir, sub_dir), exist_ok=True)
    data_path = os.path.join(input_path, "_".join(feats)) + "/"

    feat_list_id_k = expand_grid(n_feats, tree_depth)

    if runparallel:
        from multiprocessing import Pool
        args = [
            (k, feat_list_id_k, feats, data_path, tree_depth,
             y_min, y_max, cnames, main_dir, sub_dir, alpha, dead_threshold)
            for k in range(n_feats ** tree_depth)
        ]
        with Pool(paralleln) as pool:
            pool.starmap(_build_one_tree_permutation, args)
    else:
        for k in range(n_feats ** tree_depth):
            print(k + 1)
            _build_one_tree_permutation(
                k, feat_list_id_k, feats, data_path, tree_depth,
                y_min, y_max, cnames, main_dir, sub_dir, alpha, dead_threshold,
            )
