"""Unit tests for `fuzzy_tree_portfolio_helper.py`.

Design of the fixture (used throughout):

    firm   size   x1=size-qntl   x2   ret
    A      1      0.1            0.9   0.01
    B      2      0.3            0.1   0.02
    C      3      0.7            0.8   0.03
    D      4      0.9            0.2   0.04

Split char order: feat_list = ["x1", "x2"].

Hand-computed reference values (at alpha -> inf, hard median split):
    root VW return                           = (1*.01 + 2*.02 + 3*.03 + 4*.04) / 10 = 0.03
    depth-1 low  (A, B) VW return            = (1*.01 + 2*.02) / 3 = 0.05 / 3
    depth-1 high (C, D) VW return            = (3*.03 + 4*.04) / 7 = 0.25 / 7
    depth-2 low-low   (B alone)              = 0.02
    depth-2 low-high  (A alone)              = 0.01
    depth-2 high-low  (D alone)              = 0.04
    depth-2 high-high (C alone)              = 0.03

Column layout (BFS, low-first within each parent):
    col 0 = root
    col 1, 2 = depth-1 [low, high]
    col 3, 4, 5, 6 = depth-2 [low-low, low-high, high-low, high-high]

Run directly:
    python -m src.tests.test_fuzzy_tree_portfolio_helper
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd

from src.code.portfolio_creation.tree_portfolio_creation.fuzzy_tree_portfolio_helper import (
    _leaf_return,
    fuzzy_tree_month,
    fuzzy_tree_portfolio,
)


# --------------------------------------------------------------- fixture

def _fixture_df():
    return pd.DataFrame(
        {
            "mm": [1, 1, 1, 1],
            "size": [1.0, 2.0, 3.0, 4.0],
            "x1": [0.1, 0.3, 0.7, 0.9],
            "x2": [0.9, 0.1, 0.8, 0.2],
            "ret": [0.01, 0.02, 0.03, 0.04],
        }
    )


ROOT_VW = 0.30 / 10.0
LOW_VW = 0.05 / 3.0
HIGH_VW = 0.25 / 7.0


# ---------------------------------------------------------- _leaf_return

def test_leaf_return_equal_weights_sizes_is_mean():
    w = np.array([1.0, 1.0, 1.0])
    size = np.array([1.0, 1.0, 1.0])
    ret = np.array([0.01, 0.02, 0.03])
    assert _leaf_return(w, size, ret) == 0.02


def test_leaf_return_value_weighted():
    w = np.array([1.0, 1.0])
    size = np.array([1.0, 3.0])
    ret = np.array([0.10, 0.20])
    # (1*1*0.10 + 1*3*0.20) / (1 + 3) = 0.70 / 4 = 0.175
    np.testing.assert_allclose(_leaf_return(w, size, ret), 0.175)


def test_leaf_return_fuzzy_weights_scale_contribution():
    """Doubling fuzzy weight on one firm must shift the return toward it."""
    w = np.array([1.0, 1.0])
    size = np.array([1.0, 1.0])
    ret = np.array([0.0, 1.0])
    base = _leaf_return(w, size, ret)  # 0.5
    w2 = np.array([1.0, 3.0])
    heavy = _leaf_return(w2, size, ret)  # (1*0 + 3*1) / 4 = 0.75
    assert base == 0.5
    np.testing.assert_allclose(heavy, 0.75)


def test_leaf_return_zero_denominator_returns_nan():
    w = np.array([0.0, 0.0])
    size = np.array([1.0, 1.0])
    ret = np.array([0.1, 0.2])
    assert np.isnan(_leaf_return(w, size, ret))


# ---------------------------------------------------- fuzzy_tree_month depth 1

def test_depth1_three_portfolios_hard_limit():
    """Depth-1 tree = 3 nodes (root, low, high). alpha -> inf matches hard split."""
    df = _fixture_df()
    out = fuzzy_tree_month(df, feat_list=["x1"], tree_depth=1, alpha=1e6)
    assert out.shape == (3,)
    np.testing.assert_allclose(out[0], ROOT_VW, atol=1e-12)
    np.testing.assert_allclose(out[1], LOW_VW, atol=1e-10)
    np.testing.assert_allclose(out[2], HIGH_VW, atol=1e-10)


def test_depth1_root_equals_full_cross_section_always():
    """Root return is the full-cross-section VW return regardless of alpha."""
    df = _fixture_df()
    for alpha in [0.01, 0.5, 2.0, 10.0, 1e6]:
        out = fuzzy_tree_month(df, feat_list=["x1"], tree_depth=1, alpha=alpha)
        np.testing.assert_allclose(out[0], ROOT_VW, atol=1e-12)


def test_depth1_mass_weighted_decomposition():
    """r_root = (r_low * M_low + r_high * M_high) / (M_low + M_high),
    where M_child = sum over firms of w_child_i * size_i. Must hold at any alpha."""
    df = _fixture_df()
    from src.code.portfolio_creation.tree_portfolio_creation.fuzzy_tree_split import (
        fuzzy_split,
    )
    for alpha in [0.3, 1.0, 5.0, 20.0]:
        x = df["x1"].to_numpy(float)
        w0 = np.ones(len(df))
        w_hi, w_lo, _ = fuzzy_split(x, w0, alpha=alpha)
        size = df["size"].to_numpy(float)
        ret = df["ret"].to_numpy(float)
        M_lo, M_hi = (w_lo * size).sum(), (w_hi * size).sum()
        r_lo = (w_lo * size * ret).sum() / M_lo
        r_hi = (w_hi * size * ret).sum() / M_hi
        implied_root = (r_lo * M_lo + r_hi * M_hi) / (M_lo + M_hi)
        np.testing.assert_allclose(implied_root, ROOT_VW, atol=1e-12)


# ---------------------------------------------------- fuzzy_tree_month depth 2

def test_depth2_seven_portfolios_hard_limit():
    """Depth-2 tree = 7 nodes. At alpha -> inf each depth-2 leaf isolates one firm."""
    df = _fixture_df()
    out = fuzzy_tree_month(df, feat_list=["x1", "x2"], tree_depth=2, alpha=1e6)
    assert out.shape == (7,)
    np.testing.assert_allclose(out[0], ROOT_VW, atol=1e-12)
    np.testing.assert_allclose(out[1], LOW_VW, atol=1e-10)
    np.testing.assert_allclose(out[2], HIGH_VW, atol=1e-10)
    # depth-2 leaves each contain a single firm -> return = that firm's ret
    np.testing.assert_allclose(out[3], 0.02, atol=1e-10)  # low-low  -> B
    np.testing.assert_allclose(out[4], 0.01, atol=1e-10)  # low-high -> A
    np.testing.assert_allclose(out[5], 0.04, atol=1e-10)  # high-low -> D
    np.testing.assert_allclose(out[6], 0.03, atol=1e-10)  # high-high-> C


def test_depth2_alpha_zero_limit_collapses_every_leaf_to_root():
    """alpha -> 0 gives every firm w=0.5 at every split, so every node's
    per-firm weight simplifies to a constant and every leaf returns the
    full-cross-section VW return."""
    df = _fixture_df()
    out = fuzzy_tree_month(df, feat_list=["x1", "x2"], tree_depth=2, alpha=1e-8)
    np.testing.assert_allclose(out, np.full(7, ROOT_VW), atol=1e-8)


def test_depth2_conservation_across_levels():
    """Sum of (M_leaf * r_leaf) across the 4 depth-2 leaves equals
    sum_i size_i * ret_i (the raw VW numerator). Tests that the fuzzy
    weight mass + value-weight combine correctly after two splits."""
    df = _fixture_df()
    from src.code.portfolio_creation.tree_portfolio_creation.fuzzy_tree_split import (
        fuzzy_split,
    )
    # Reconstruct leaf weights manually to get M_leaf = sum(w_leaf * size)
    alpha = 4.0
    x1 = df["x1"].to_numpy(float)
    x2 = df["x2"].to_numpy(float)
    size = df["size"].to_numpy(float)
    ret = df["ret"].to_numpy(float)
    w0 = np.ones(len(df))
    w_hi1, w_lo1, _ = fuzzy_split(x1, w0, alpha=alpha)
    leaves = []
    for parent in (w_lo1, w_hi1):
        w_hi2, w_lo2, _ = fuzzy_split(x2, parent, alpha=alpha)
        leaves.append(w_lo2)
        leaves.append(w_hi2)

    out = fuzzy_tree_month(df, feat_list=["x1", "x2"], tree_depth=2, alpha=alpha)
    leaf_returns = out[3:7]

    numerator = 0.0
    denominator = 0.0
    for w_leaf, r_leaf in zip(leaves, leaf_returns):
        M = (w_leaf * size).sum()
        numerator += M * r_leaf
        denominator += M
    np.testing.assert_allclose(numerator / denominator, ROOT_VW, atol=1e-12)
    # And sum of leaf masses must equal sum of sizes (no mass created/destroyed).
    np.testing.assert_allclose(denominator, size.sum(), atol=1e-12)


# ---------------------------------------------------- fuzzy_tree_portfolio I/O

def test_fuzzy_tree_portfolio_reads_yearly_chunks_and_returns_shape():
    """End-to-end: write 2 yearly CSVs, run the sweep, check shape + root column."""
    df1 = _fixture_df().copy()
    df2 = _fixture_df().copy()
    df2["ret"] = df2["ret"] * 2.0  # different year, different returns

    with tempfile.TemporaryDirectory() as d:
        df1.to_csv(os.path.join(d, "toy_2001.csv"), index=False)
        df2.to_csv(os.path.join(d, "toy_2002.csv"), index=False)
        out = fuzzy_tree_portfolio(
            data_path=d + os.sep,
            feat_list=["x1", "x2"],
            tree_depth=2,
            y_min=2001,
            y_max=2002,
            file_prefix="toy_",
            alpha=1e6,
        )

    assert out.shape == (24, 7)  # 2 years * 12 months, 2**(2+1) - 1 cols
    # Month 1 of year 1: root = 0.03; every other month: NaN (fixture only has mm=1).
    np.testing.assert_allclose(out[0, 0], ROOT_VW, atol=1e-12)
    assert np.all(np.isnan(out[1, :]))
    # Month 1 of year 2: returns doubled -> root = 0.06
    np.testing.assert_allclose(out[12, 0], 2 * ROOT_VW, atol=1e-12)


# ------------------------------------------------------------------- runner

def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failures.append((name, e))
            print(f"  FAIL  {name}: {e!r}")
    print()
    print(f"{len(tests) - len(failures)} / {len(tests)} passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_all()
