"""
Elastic-net regression via the LARS augmented-matrix trick.

Mirrors `reference_code/2_AP_Pruning/lasso.R`:

    yy = c(y, rep(0, p))
    XX = rbind(X, diag(sqrt(lambda2), p, p))
    lasso_obj = lars(XX, yy, type="lasso", normalize=FALSE, intercept=FALSE)
    beta = coef(lasso_obj)
    K = apply(beta, 1, function(x) sum(x != 0))
    subset = K >= kmin & K <= kmax
    return list(beta[subset,], K[subset])

Appending sqrt(lambda2)*I to X and zero-padding y converts an L1+L2 problem
to a pure L1 problem solvable by LARS. Uses sklearn's `lars_path`.
"""

import numpy as np
from sklearn.linear_model import lars_path


def lasso_en(X, y, lambda2, kmin=5, kmax=50):
    """
    Return (betas, K) where
        betas: array of shape (n_kept_steps, p) -- LARS-path coefficients
        K:     array of shape (n_kept_steps,)   -- nonzero count per step

    Only steps with K in [kmin, kmax] are retained, matching lasso.R.
    """
    n, p = X.shape
    XX = np.vstack([X, np.sqrt(lambda2) * np.eye(p)])
    yy = np.concatenate([y, np.zeros(p)])

    # lars_path returns coefs of shape (p, n_steps). method='lasso' matches R's
    # lars(type='lasso'). We pass un-normalized X to match normalize=FALSE.
    _, _, coefs = lars_path(XX, yy, method="lasso")
    # Transpose to (n_steps, p) so each row is one step's beta vector.
    betas = coefs.T

    K = (betas != 0).sum(axis=1)
    mask = (K >= kmin) & (K <= kmax)
    return betas[mask], K[mask]
