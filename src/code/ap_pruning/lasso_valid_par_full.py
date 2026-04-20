"""Python translation of `reference_code/2_AP_Pruning/lasso_valid_par_full.R`.

Elastic-net grid search with cross-validation: for each (lambda0, lambda2)
pair, run the LARS-EN path, compute train/valid/test Sharpe ratios over the
returned K grid, and write one CSV per (cv_fold, lambda0_idx, lambda2_idx).
"""

import os

import numpy as np
import pandas as pd

from src.code.ap_pruning.lasso import lasso


def lasso_valid_full(ports, lambda0, lambda2, main_dir, sub_dir, adj_w,
                     n_train_valid=360, cvN=3, runFullCV=False,
                     kmin=5, kmax=50, RunParallel=False, ParallelN=10):
    os.makedirs(os.path.join(main_dir, sub_dir), exist_ok=True)

    ports = np.asarray(ports, dtype=float)
    # R: ports[(n_train_valid + 1):(nrow(ports)), ]
    ports_test = ports[n_train_valid:, :]

    n_valid = n_train_valid // cvN
    n_train = n_train_valid - n_valid  # noqa: F841 — matches R (declared but unused downstream)

    if runFullCV:
        cv_range = range(1, cvN + 1)
    else:
        cv_range = range(cvN, cvN + 1)  # R: cvN:cvN — only the last fold

    for i in cv_range:
        # R: ports[-c(((i-1)*n_valid+1):(i*n_valid), (n_train_valid+1):(nrow)), ]
        # = exclude both the validation fold and the test window
        train_mask = np.ones(ports.shape[0], dtype=bool)
        train_mask[(i - 1) * n_valid:i * n_valid] = False
        train_mask[n_train_valid:] = False
        ports_train = ports[train_mask, :]
        ports_valid = ports[(i - 1) * n_valid:i * n_valid, :]
        lasso_cv_helper(
            ports_train, ports_valid, ports_test, lambda0, lambda2,
            main_dir, sub_dir, adj_w, ports.shape[0],
            f"cv_{i}", kmin, kmax, RunParallel, ParallelN,
        )

    # After pinning down the parameter, do another fit on the whole
    # train+valid time period.
    ports_train = ports[:n_train_valid, :]
    lasso_cv_helper(
        ports_train, None, ports_test, lambda0, lambda2,
        main_dir, sub_dir, adj_w, ports.shape[0],
        "full", kmin, kmax, RunParallel, ParallelN,
    )


def _process_lambda0(i, lambda0, lambda2, sigma_tilde, mu_tilde,
                     ports_train, ports_valid, ports_test, adj_w,
                     mu_len, col_names, main_dir, sub_dir, cv_name,
                     kmin, kmax):
    """Inner per-lambda0 loop body (used by both serial and parallel paths)."""
    for j in range(len(lambda2)):
        lasso_results = lasso(
            sigma_tilde, mu_tilde[:, i], lambda2[j], 100, kmin, kmax
        )
        beta_mat = lasso_results[0]
        portsN = lasso_results[1]
        R = beta_mat.shape[0]

        train_SR = np.zeros(R)
        valid_SR = np.zeros(R) if ports_valid is not None else None
        test_SR = np.zeros(R)
        betas = np.full((R, mu_len), np.nan)

        for r in range(R):
            b = beta_mat[r, :].copy()
            b = b * adj_w
            b = b / abs(np.sum(b))

            sdf_train = ports_train @ (b / adj_w)
            sdf_test = ports_test @ (b / adj_w)

            # R: mean(x)/sd(x); sd uses n-1 denominator, so ddof=1.
            train_SR[r] = np.mean(sdf_train) / np.std(sdf_train, ddof=1)
            if ports_valid is not None:
                sdf_valid = ports_valid @ (b / adj_w)
                valid_SR[r] = np.mean(sdf_valid) / np.std(sdf_valid, ddof=1)
            test_SR[r] = np.mean(sdf_test) / np.std(sdf_test, ddof=1)
            betas[r, :] = b

        beta_df = pd.DataFrame(betas, columns=col_names)
        if ports_valid is not None:
            results = pd.concat([
                pd.DataFrame({"train_SR": train_SR, "valid_SR": valid_SR,
                              "test_SR": test_SR, "portsN": portsN}),
                beta_df,
            ], axis=1)
        else:
            results = pd.concat([
                pd.DataFrame({"train_SR": train_SR, "test_SR": test_SR,
                              "portsN": portsN}),
                beta_df,
            ], axis=1)

        # R: paste(..., 'results_',cv_name,'_l0_', i,'_l2_',j,'.csv', sep='')
        # R uses 1-indexed i, j; we emit 1-indexed for file-name parity.
        out = os.path.join(
            main_dir, sub_dir,
            f"results_{cv_name}_l0_{i + 1}_l2_{j + 1}.csv",
        )
        results.to_csv(out, index=False)


def lasso_cv_helper(ports_train, ports_valid, ports_test, lambda0, lambda2,
                    main_dir, sub_dir, adj_w, n_total, cv_name,
                    kmin=5, kmax=50, RunParallel=False, ParallelN=10):
    ports_train = np.asarray(ports_train, dtype=float)
    ports_test = np.asarray(ports_test, dtype=float)
    if ports_valid is not None:
        ports_valid = np.asarray(ports_valid, dtype=float)
    adj_w = np.asarray(adj_w, dtype=float)
    lambda0 = np.asarray(lambda0, dtype=float).ravel()
    lambda2 = np.asarray(lambda2, dtype=float).ravel()

    # Converting the optimization into a regression problem
    # R: mu = apply(ports_train, 2, mean); sigma = cov(ports_train)
    mu = ports_train.mean(axis=0)
    sigma = np.cov(ports_train, rowvar=False)  # n-1 denom matches R cov()

    mu_bar = mu.mean()
    gamma = min(ports_train.shape)
    # R: eigen() returns values in decreasing order; numpy eigh returns ascending.
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    gamma = min(gamma, int(np.sum(eigvals > 1e-10)))
    D = eigvals[:gamma]
    V = eigvecs[:, :gamma]

    sigma_tilde = V @ np.diag(np.sqrt(D)) @ V.T

    # R:
    #   matrix(rep(mu, length(lambda0)), nrow=length(mu), ncol=length(lambda0))
    #   + matrix(rep(lambda0, each=length(mu)) * mu_bar, nrow=length(mu), ncol=length(lambda0))
    # = mu replicated across cols + lambda0*mu_bar replicated across rows.
    n_lam0 = len(lambda0)
    mu_len = len(mu)
    mu_expand = np.tile(mu[:, None], (1, n_lam0))
    lam0_term = np.ones((mu_len, 1)) * (lambda0 * mu_bar)[None, :]
    mu_adj = mu_expand + lam0_term  # (mu_len, n_lam0)

    mu_tilde = V @ np.diag(1 / np.sqrt(D)) @ V.T @ mu_adj
    # w_tilde declared in R but unused elsewhere — keep the computation for parity.
    w_tilde = V @ np.diag(1 / D) @ V.T @ mu_adj  # noqa: F841

    col_names = getattr(ports_train, "columns", None)
    if col_names is None:
        col_names = [f"V{k + 1}" for k in range(mu_len)]
    else:
        col_names = list(col_names)

    if RunParallel:
        from multiprocessing import Pool
        args = [
            (i, lambda0, lambda2, sigma_tilde, mu_tilde,
             ports_train, ports_valid, ports_test, adj_w, mu_len, col_names,
             main_dir, sub_dir, cv_name, kmin, kmax)
            for i in range(n_lam0)
        ]
        with Pool(ParallelN) as pool:
            pool.starmap(_process_lambda0, args)
    else:
        for i in range(n_lam0):
            _process_lambda0(
                i, lambda0, lambda2, sigma_tilde, mu_tilde,
                ports_train, ports_valid, ports_test, adj_w, mu_len, col_names,
                main_dir, sub_dir, cv_name, kmin, kmax,
            )
