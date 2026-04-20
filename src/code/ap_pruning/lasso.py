"""Python translation of `reference_code/2_AP_Pruning/lasso.R`.

Call LARS to calculate the whole path for EN regularized regression.
Uses sklearn's `lars_path` as the Python analogue of R's `lars` package
(R: library(lars); Python: sklearn.linear_model.lars_path).
"""

import numpy as np
from sklearn.linear_model import lars_path


def lasso(X, y, lambda2, steps=70, kmin=5, kmax=50):
    """Call LARS to calculate the whole path for EN regularized regression.

    Uses the augmented-matrix trick: augment X with sqrt(lambda2)*I and y
    with zeros, then solve lasso.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = X.shape[0]
    p = X.shape[1]
    yy = np.concatenate([y, np.zeros(p)])
    XX = np.vstack([X, np.sqrt(lambda2) * np.eye(p)])

    # R: lars(XX, yy, type="lasso", normalize=FALSE, intercept=FALSE)
    # sklearn.lars_path with method="lasso"; no normalize/center by default.
    _, _, coefs = lars_path(XX, yy, method="lasso")
    # coefs is (p, n_alphas). R's coef(lars_obj) is (n_alphas, p). Transpose.
    beta = coefs.T

    K = np.array([int(np.sum(row != 0)) for row in beta])
    subset = (K >= kmin) & (K <= kmax)
    return [beta[subset, :], K[subset]]
