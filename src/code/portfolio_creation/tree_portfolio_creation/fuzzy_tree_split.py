"""Fuzzy (soft) split primitive for the AP-Tree ablation.

Pure numerical kernel only — no I/O, no pandas. Takes arrays in, returns
arrays out. See `docs/abalation_1.md` for the design.

Split rule (per-firm sigmoid centered at the within-parent median):
    m     = weighted_median(x, w_parent)
    s_i   = sigmoid(alpha * (x_i - m))
    w_hi  = s_i       * w_parent_i     (top / "high" child)
    w_lo  = (1 - s_i) * w_parent_i     (bottom / "low"  child)

Guarantees:
    per-firm conservation: w_hi + w_lo == w_parent exactly
    sibling mass balance:  sum(w_hi) ~= sum(w_lo) ~= sum(w_parent)/2
    alpha -> inf:          recovers the hard median split (1{x > m})
    alpha -> 0:            all firms split 50/50 (degenerate sanity check)
"""

from __future__ import annotations

import numpy as np


def sigmoid(u: np.ndarray) -> np.ndarray:
    """Numerically-stable logistic sigmoid, elementwise."""
    u = np.asarray(u, dtype=np.float64)
    out = np.empty_like(u)
    pos = u >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-u[pos]))
    ex = np.exp(u[neg])
    out[neg] = ex / (1.0 + ex)
    return out


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    """Weighted median over the nonzero-weight support.

    Zero-weight firms are dropped first — a firm with w == 0 has taken
    the other branch upstream (happens at α → ∞) and should not shift
    the threshold. Among the remaining firms, returns the smallest x
    whose cumulative weight reaches total/2. When the cumulative weight
    lands exactly on the half mark between two firms, returns the
    average of the two straddling values (matches `np.median` on equal
    weights).
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if x.size == 0:
        raise ValueError("weighted_median: empty input")
    if np.any(w < 0):
        raise ValueError("weighted_median: negative weights not allowed")
    mask = w > 0
    if not np.any(mask):
        raise ValueError("weighted_median: total weight must be positive")
    x = x[mask]
    w = w[mask]

    order = np.argsort(x, kind="mergesort")
    x_s = x[order]
    w_s = w[order]
    cw = np.cumsum(w_s)
    total = float(cw[-1])
    half = 0.5 * total

    k = int(np.searchsorted(cw, half, side="left"))
    # Straddle: cw[k] sits exactly on the half mark (within float tolerance)
    # AND there is a firm after k — return the midpoint with x_s[k+1].
    tol = 1e-12 * max(total, 1.0)
    if k + 1 < x_s.size and abs(cw[k] - half) <= tol:
        return 0.5 * (x_s[k] + x_s[k + 1])
    return float(x_s[k])


def fuzzy_split(
    x: np.ndarray,
    w_parent: np.ndarray,
    alpha: float,
    m: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Soft split of a parent node into (high, low) children.

    Parameters
    ----------
    x : (n,) array
        Quantile-normalized characteristic values for the firms in the
        parent node (already in [0, 1] from step 1 of the pipeline).
    w_parent : (n,) array
        Parent-node membership weights. Root: all ones.
    alpha : float, > 0
        Sigmoid steepness. alpha -> inf recovers the hard median split.
    m : float, optional
        Split threshold. If None, computed as weighted_median(x, w_parent).
        Caller can pass a fixed m for tests or hard-split validation.

    Returns
    -------
    w_high : (n,) array
        Weights assigned to the top / "high" child.
    w_low : (n,) array
        Weights assigned to the bottom / "low" child.
    m : float
        The threshold used (useful for logging / tests).
    """
    x = np.asarray(x, dtype=np.float64)
    w_parent = np.asarray(w_parent, dtype=np.float64)
    if x.shape != w_parent.shape:
        raise ValueError(
            f"fuzzy_split: shape mismatch x={x.shape} w_parent={w_parent.shape}"
        )
    if x.ndim != 1:
        raise ValueError(f"fuzzy_split: expected 1-D arrays, got ndim={x.ndim}")
    if alpha <= 0:
        raise ValueError(f"fuzzy_split: alpha must be positive, got {alpha}")

    if m is None:
        m = weighted_median(x, w_parent)

    s = sigmoid(alpha * (x - m))
    w_high = s * w_parent
    w_low = w_parent - w_high  # exact conservation (no (1-s) roundoff)
    return w_high, w_low, m
