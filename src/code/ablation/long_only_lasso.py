"""
Long-only variant of `lasso_en`: LARS-EN with a non-negativity constraint on
the coefficient path.

Mirrors `src/code/ap_pruning/lasso.py` one-to-one except for `positive=True`
in the underlying `lars_path` call. Baseline `lasso_en` stays unchanged so
Phase D / Phase F results are bit-for-bit preserved; this module is invoked
only by the long-only ablation runner.

Reference: Ablation #2 in `docs/abalation_planning.md`
(economic implementability — SDF that cannot short leaf portfolios).
"""

import numpy as np
from sklearn.linear_model import lars_path


def lasso_en_positive(X, y, lambda2, kmin=5, kmax=50):
    """
    Non-negative LARS-EN path on the augmented (X, y) system.

    Returns (betas, K) with the same shape conventions as `lasso_en`:
      betas : array (n_kept_steps, p) -- coefficient vectors along the path
      K     : array (n_kept_steps,)   -- nonzero count per step

    Only steps with K in [kmin, kmax] are retained. All retained β rows
    satisfy β_i ≥ 0 by construction (enforced inside LARS).
    """
    n, p = X.shape
    XX = np.vstack([X, np.sqrt(lambda2) * np.eye(p)])
    yy = np.concatenate([y, np.zeros(p)])

    _, _, coefs = lars_path(XX, yy, method="lasso", positive=True)
    betas = coefs.T

    K = (betas != 0).sum(axis=1)
    mask = (K >= kmin) & (K <= kmax)
    return betas[mask], K[mask]
