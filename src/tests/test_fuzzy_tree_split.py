"""Unit tests for `fuzzy_tree_split.py` primitives.

Run directly:
    python -m src.tests.test_fuzzy_tree_split
"""

from __future__ import annotations

import numpy as np

from src.code.portfolio_creation.tree_portfolio_creation.fuzzy_tree_split import (
    fuzzy_split,
    sigmoid,
    weighted_median,
)


# ------------------------------------------------------------------ sigmoid

def test_sigmoid_basic_values():
    u = np.array([0.0, 1.0, -1.0])
    s = sigmoid(u)
    np.testing.assert_allclose(s, [0.5, 1 / (1 + np.exp(-1)), 1 / (1 + np.exp(1))])


def test_sigmoid_numerical_stability_large_inputs():
    # Should not overflow and should saturate to the correct side.
    s = sigmoid(np.array([1e3, -1e3, 1e8, -1e8]))
    np.testing.assert_allclose(s, [1.0, 0.0, 1.0, 0.0], atol=1e-15)


def test_sigmoid_bounds():
    rng = np.random.default_rng(0)
    u = rng.standard_normal(1000) * 50
    s = sigmoid(u)
    assert np.all((s >= 0.0) & (s <= 1.0))


# ------------------------------------------------------------- weighted_median

def test_weighted_median_unweighted_matches_numpy_odd():
    x = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    w = np.ones_like(x)
    assert weighted_median(x, w) == np.median(x)  # 3.0


def test_weighted_median_unweighted_matches_numpy_even():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.ones_like(x)
    # cumulative weight reaches exactly half at x=2 -> average with x=3 -> 2.5
    assert weighted_median(x, w) == np.median(x)  # 2.5


def test_weighted_median_shifts_toward_heavy_firm():
    # Firm at x=3 has weight 10, total=12, half=6. cumsum=[1,2,12].
    # First index where cw >= 6 is 2 -> median = 3.
    x = np.array([1.0, 2.0, 3.0])
    w = np.array([1.0, 1.0, 10.0])
    assert weighted_median(x, w) == 3.0


def test_weighted_median_shifts_other_direction():
    x = np.array([1.0, 2.0, 3.0])
    w = np.array([10.0, 1.0, 1.0])
    # total=12, half=6, cumsum=[10,11,12]. First idx where >= 6 is 0 -> median = 1.
    assert weighted_median(x, w) == 1.0


def test_weighted_median_rejects_bad_input():
    try:
        weighted_median(np.array([]), np.array([]))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on empty input")

    try:
        weighted_median(np.array([1.0, 2.0]), np.array([-1.0, 1.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on negative weight")

    try:
        weighted_median(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on zero total weight")


# ---------------------------------------------------------------- fuzzy_split

def test_fuzzy_split_conservation():
    """w_high + w_low must equal w_parent to rounding tolerance.

    Exact equality is not achievable in IEEE float because w_low is
    computed as w_parent - w_high; adding w_high back can drift by
    <= 1 ULP. A few ULPs of tolerance is the right bar.
    """
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, size=100)
    w_parent = rng.uniform(0.01, 2.0, size=100)
    for alpha in [0.1, 1.0, 10.0, 100.0, 1e6]:
        w_hi, w_lo, _ = fuzzy_split(x, w_parent, alpha=alpha)
        np.testing.assert_allclose(w_hi + w_lo, w_parent, rtol=0, atol=1e-15)


def test_fuzzy_split_sibling_mass_balanced_at_median():
    """With m at the weighted median, sibling masses should be ~ equal."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, size=1000)
    w_parent = np.ones_like(x)
    w_hi, w_lo, m = fuzzy_split(x, w_parent, alpha=10.0)
    total = w_parent.sum()
    # within a few % of perfect balance on a large sample
    assert abs(w_hi.sum() - total / 2) / total < 0.02
    assert abs(w_lo.sum() - total / 2) / total < 0.02


def test_fuzzy_split_hard_limit_large_alpha():
    """alpha -> inf must recover the hard median split."""
    x = np.array([0.1, 0.3, 0.7, 0.9])
    w_parent = np.ones_like(x)
    w_hi, w_lo, m = fuzzy_split(x, w_parent, alpha=1e6)
    # m = 0.5 (median of 4 evenly spaced points straddling 0.5)
    assert abs(m - 0.5) < 1e-12
    np.testing.assert_allclose(w_hi, [0.0, 0.0, 1.0, 1.0], atol=1e-10)
    np.testing.assert_allclose(w_lo, [1.0, 1.0, 0.0, 0.0], atol=1e-10)


def test_fuzzy_split_soft_limit_small_alpha():
    """alpha -> 0 must produce a 50/50 split for every firm."""
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    w_parent = np.ones_like(x)
    w_hi, w_lo, _ = fuzzy_split(x, w_parent, alpha=1e-8)
    np.testing.assert_allclose(w_hi, 0.5 * w_parent, atol=1e-7)
    np.testing.assert_allclose(w_lo, 0.5 * w_parent, atol=1e-7)


def test_fuzzy_split_parent_weight_scales_output():
    """Scaling w_parent by c must scale both children by c."""
    x = np.array([0.1, 0.4, 0.6, 0.9])
    w1 = np.ones_like(x)
    w2 = 3.0 * w1
    hi1, lo1, m1 = fuzzy_split(x, w1, alpha=5.0)
    hi2, lo2, m2 = fuzzy_split(x, w2, alpha=5.0)
    # weighted median unchanged under uniform rescaling
    assert m1 == m2
    np.testing.assert_allclose(hi2, 3.0 * hi1)
    np.testing.assert_allclose(lo2, 3.0 * lo1)


def test_fuzzy_split_explicit_m_overrides_default():
    """Passing m bypasses the weighted-median computation."""
    x = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    w = np.ones_like(x)
    w_hi, w_lo, m_used = fuzzy_split(x, w, alpha=1e6, m=0.3)
    assert m_used == 0.3
    # hard limit with threshold 0.3: x <= 0.3 -> low, x > 0.3 -> high
    np.testing.assert_allclose(w_hi, [0.0, 0.0, 1.0, 1.0, 1.0, 1.0], atol=1e-10)
    np.testing.assert_allclose(w_lo, [1.0, 1.0, 0.0, 0.0, 0.0, 0.0], atol=1e-10)


def test_fuzzy_split_rejects_bad_input():
    x = np.array([0.1, 0.5, 0.9])
    w = np.ones(3)
    for bad_alpha in [0.0, -1.0]:
        try:
            fuzzy_split(x, w, alpha=bad_alpha)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for alpha={bad_alpha}")

    try:
        fuzzy_split(x, np.ones(4), alpha=1.0)  # shape mismatch
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on shape mismatch")


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
