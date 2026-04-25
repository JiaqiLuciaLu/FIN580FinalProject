"""Fuzzy (soft-split) analogue of `tree_portfolio_helper.py`.

Builds AP-Tree portfolios at every depth level, per month, using the
per-firm sigmoid split in `fuzzy_tree_split.fuzzy_split`. Unlike the
hard-split helper — where each firm is assigned to exactly one leaf per
level via `ntile` — here every node carries a weight *vector* over the
firms in the cross-section, and weights propagate multiplicatively from
root to leaf.

Output layout (`ret_table`) matches the hard-split helper so the rest of
the pipeline (step3 rmrf, step4 monoculture filter, pruning, metrics)
can consume the output unchanged:

    shape  : (n_months, 2**(tree_depth + 1) - 1)
    columns: breadth-first (level-major, low-then-high within each
             parent) ordering. For depth i, node k in 1..2**i,
             col = 2**i - 1 + (k - 1). k=1 is the all-low path,
             k=2**i is the all-high path.

v1 scope (see `docs/abalation_1.md`): ret_table only — no per-leaf
`feat_min` / `feat_max` tables (meaningless when every firm is in every
leaf; defer to weighted-quantile follow-up).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.code.portfolio_creation.tree_portfolio_creation.fuzzy_tree_split import (
    fuzzy_split,
)


def _leaf_return(
    w: np.ndarray,
    size: np.ndarray,
    ret: np.ndarray,
    dead_threshold: float = 0.0,
) -> float:
    """Value- *and* fuzzy-weighted return, renormalized so leaf weights sum to 1.

    r_L = Σ (w_i · size_i · ret_i) / Σ (w_i · size_i)

    If `dead_threshold > 0`, firms with `w_i < dead_threshold` are zeroed out
    before the weighted average. The renormalization in the denominator then
    rescales the surviving firms back to sum-to-1 — i.e. the dead-stock tail
    is dropped, not just downweighted. Used for the overfitting-vs-pruning
    diagnostic on fuzzy α=50.
    """
    if dead_threshold > 0.0:
        w = np.where(w < dead_threshold, 0.0, w)
    ws = w * size
    den = ws.sum()
    if den <= 0 or not np.isfinite(den):
        return np.nan
    return float(np.dot(ws, ret) / den)


def fuzzy_tree_month(
    df_m: pd.DataFrame,
    feat_list: list[str],
    tree_depth: int,
    alpha: float,
    dead_threshold: float = 0.0,
) -> np.ndarray:
    """Build the full fuzzy tree for one (year, month) cross-section.

    Parameters
    ----------
    df_m : DataFrame
        Rows = firms in this month; must contain `ret`, `size`, and every
        name in `feat_list`. Characteristic columns are the quantile-
        normalized values produced by step 1.
    feat_list : list[str]
        Split characteristic at each depth, length == `tree_depth`.
        `feat_list[i]` splits level-i parents into level-(i+1) children.
    tree_depth : int
        Number of splits (depth-d tree has 2**d leaves, 2**(d+1)-1 nodes).
    alpha : float
        Sigmoid steepness. Large alpha → hard median split.

    Returns
    -------
    (2**(tree_depth+1) - 1,) float array of portfolio returns, one per
    node, in the column order described in this module's docstring.
    """
    if len(feat_list) < tree_depth:
        raise ValueError(
            f"feat_list too short: {len(feat_list)} < tree_depth={tree_depth}"
        )

    n = len(df_m)
    size = df_m["size"].to_numpy(dtype=np.float64)
    ret = df_m["ret"].to_numpy(dtype=np.float64)

    # BFS: levels[i] = list of weight vectors for the 2**i nodes at depth i.
    # Root: every firm has weight 1.
    levels: list[list[np.ndarray]] = [[np.ones(n, dtype=np.float64)]]
    for i in range(tree_depth):
        x = df_m[feat_list[i]].to_numpy(dtype=np.float64)
        next_level: list[np.ndarray] = []
        for w_parent in levels[i]:
            w_high, w_low, _ = fuzzy_split(x, w_parent, alpha=alpha)
            # Match ntile convention: k=1 is the LOW child, k=2 is HIGH,
            # so at every parent we append (low, high) in that order.
            next_level.append(w_low)
            next_level.append(w_high)
        levels.append(next_level)

    n_cols = 2 ** (tree_depth + 1) - 1
    out = np.empty(n_cols, dtype=np.float64)
    col = 0
    for level in levels:
        for w in level:
            out[col] = _leaf_return(w, size, ret, dead_threshold=dead_threshold)
            col += 1
    return out


def fuzzy_tree_portfolio(
    data_path: str,
    feat_list: list[str],
    tree_depth: int,
    y_min: int,
    y_max: int,
    file_prefix: str,
    alpha: float,
    dead_threshold: float = 0.0,
) -> np.ndarray:
    """Sweep yearly chunks and build fuzzy-tree returns for all months.

    Mirrors `tree_portfolio_helper.tree_portfolio`'s reading pattern:
    one CSV per year at `{data_path}{file_prefix}{y}.csv` with columns
    `mm`, `ret`, `size`, and the characteristic columns named in
    `feat_list`.

    Returns
    -------
    ret_table : ((y_max-y_min+1)*12, 2**(tree_depth+1)-1) float array
        NaN rows for (year, month) combinations with no firms.
    """
    n_cols = 2 ** (tree_depth + 1) - 1
    n_rows = (y_max - y_min + 1) * 12
    ret_table = np.full((n_rows, n_cols), np.nan, dtype=np.float64)

    for y in range(y_min, y_max + 1):
        if y % 5 == 0:
            print(y)
        df_y = pd.read_csv(f"{data_path}{file_prefix}{y}.csv")
        for m in range(1, 13):
            df_m = df_y.loc[df_y["mm"] == m, :]
            if len(df_m) == 0:
                continue
            row = 12 * (y - y_min) + m - 1
            ret_table[row, :] = fuzzy_tree_month(
                df_m, feat_list, tree_depth, alpha, dead_threshold=dead_threshold,
            )

    return ret_table
