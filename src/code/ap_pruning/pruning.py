"""
AP-Pruning: grid search over (lambda0, lambda2) computing per-sparsity Sharpe ratios
on SDF portfolios constructed by LARS-EN on transformed (sigma_tilde, mu_tilde) regression.

Mirrors `reference_code/2_AP_Pruning/AP_Pruning.R` and
        `reference_code/2_AP_Pruning/lasso_valid_par_full.R`.

Pipeline (per R's main_simplified.R):
  - Extract tree depths from column names, compute adj_w = 1/sqrt(2^depth).
  - Scale columns of `ports` by adj_w to form adj_ports.
  - Split rows: train+valid = rows [0:360], test = [360:].
  - CV fold i holds out rows [(i-1)*120 : i*120]; runFullCV=False uses only
    fold cvN = 3. Always also run a 'full' fit on the full train+valid window.
  - For each (lambda0, lambda2), eigendecompose Sigma, build sigma_tilde and mu_tilde,
    run LARS-EN, compute train/valid/test SRs per sparsity level, write a CSV.
"""

import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .lasso import lasso_en


# ----------------------------------------------------------------------
# Module-level storage used to share fold arrays across multiprocessing
# workers on fork-based systems (Linux). The parent populates these once
# per fold and forks the Pool; children inherit via COW memory.
# ----------------------------------------------------------------------
_WORKER_CTX = {}


def _process_config(task):
    """Worker: compute one (i, j) grid cell and write its CSV."""
    i, j, l0, l2 = task
    c = _WORKER_CTX
    df = _run_one_config(
        c["ports_train"], c["ports_valid"], c["ports_test"],
        l0, l2, c["adj_w"], c["col_names"], c["kmin"], c["kmax"],
    )
    fname = f"results_{c['cv_name']}_l0_{i}_l2_{j}.csv"
    df.to_csv(os.path.join(c["output_dir"], fname), index=False)
    return fname


def _rename_tree_cols(cols):
    """
    Apply R's column-rename in AP_Pruning.R:13-16 exactly.

    Example: 'X1111.11' -> '1.11'; 'X2333.12222' -> '2333.12222'; 'X1111.1' -> '.1'.
    """
    out = []
    for c in cols:
        body = c[1:] if c.startswith("X") else c
        tree_id, path = body.split(".")
        prefix_len = len(path) - 1
        out.append(f"{tree_id[:prefix_len]}.{path}")
    return out


def compute_adj_w(cols):
    """adj_w_i = 1 / sqrt(2^depth_i) where depth_i = len(col_i) - 7."""
    depths = np.array([len(c) - 7 for c in cols], dtype=int)
    return 1.0 / np.sqrt(2.0 ** depths)


def _build_sigma_tilde_mu_tilde(ports_train, lambda0):
    """
    Mirror eigendecomp and construction in lasso_valid_par_full.R:35-48.

    Returns sigma_tilde (p x p) and mu_tilde (p,) for the given lambda0.
    """
    mu = ports_train.mean(axis=0)
    sigma = np.cov(ports_train, rowvar=False)

    mu_bar = mu.mean()

    # numpy eigh -> ascending; R's eigen -> descending. Sort descending to match.
    eigvals, eigvecs = np.linalg.eigh(sigma)
    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # gamma = min(min(T, N), sum(eigvals > 1e-10))
    gamma = min(min(ports_train.shape), int((eigvals > 1e-10).sum()))
    D = eigvals[:gamma]
    V = eigvecs[:, :gamma]

    sqrtD = np.sqrt(D)
    invSqrtD = 1.0 / sqrtD

    sigma_tilde = V @ np.diag(sqrtD) @ V.T
    mu_tilde = V @ np.diag(invSqrtD) @ V.T @ (mu + lambda0 * mu_bar)
    return sigma_tilde, mu_tilde


def _run_one_config(
    ports_train,
    ports_valid,
    ports_test,
    lambda0,
    lambda2,
    adj_w,
    col_names,
    kmin,
    kmax,
):
    """
    Reproduce the inner loop body of lasso_cv_helper:
      - Build sigma_tilde, mu_tilde for this lambda0.
      - Run LARS-EN at this lambda2.
      - For each retained step: denormalize betas, normalize by |sum|,
        compute train/(valid)/test SDF Sharpe ratios.

    Returns a DataFrame matching the R CSV schema:
      CV:    [train_SR, valid_SR, test_SR, portsN, <beta cols>]
      Full:  [train_SR, test_SR, portsN, <beta cols>]
    """
    sigma_tilde, mu_tilde = _build_sigma_tilde_mu_tilde(ports_train, lambda0)
    betas, K = lasso_en(sigma_tilde, mu_tilde, lambda2, kmin=kmin, kmax=kmax)

    n_steps = betas.shape[0]
    p = betas.shape[1]

    train_SR = np.empty(n_steps)
    test_SR = np.empty(n_steps)
    valid_SR = np.empty(n_steps) if ports_valid is not None else None
    out_betas = np.empty_like(betas)

    for r in range(n_steps):
        b = betas[r] * adj_w
        s = abs(b.sum())
        if s == 0:
            out_betas[r] = b
            train_SR[r] = np.nan
            if valid_SR is not None:
                valid_SR[r] = np.nan
            test_SR[r] = np.nan
            continue
        b = b / s

        # SDF = ports @ (b / adj_w) -- ports here are the *scaled* adj_ports.
        sdf_train = ports_train @ (b / adj_w)
        sdf_test = ports_test @ (b / adj_w)

        train_SR[r] = sdf_train.mean() / sdf_train.std(ddof=1)
        test_SR[r] = sdf_test.mean() / sdf_test.std(ddof=1)
        if ports_valid is not None:
            sdf_valid = ports_valid @ (b / adj_w)
            valid_SR[r] = sdf_valid.mean() / sdf_valid.std(ddof=1)

        out_betas[r] = b

    cols_header = ["train_SR"]
    arrs = [train_SR]
    if valid_SR is not None:
        cols_header.append("valid_SR")
        arrs.append(valid_SR)
    cols_header += ["test_SR", "portsN"]
    arrs += [test_SR, K]

    df_meta = pd.DataFrame(np.column_stack(arrs), columns=cols_header)
    df_meta["portsN"] = df_meta["portsN"].astype(int)
    df_betas = pd.DataFrame(out_betas, columns=col_names)
    return pd.concat([df_meta, df_betas], axis=1)


def ap_pruning(
    ports,
    lambda0_list,
    lambda2_list,
    output_dir,
    n_train_valid=360,
    cv_n=3,
    run_full_cv=False,
    kmin=5,
    kmax=50,
    is_tree=True,
    n_workers=1,
):
    """
    Top-level entry point analogous to AP_Pruning() -> lasso_valid_full() in R.

    `ports` is a DataFrame (T x N) with tree-encoded column names.
    Writes one CSV per (cv_name, i, j) combination into `output_dir`:
        results_{cv_name}_l0_{i}_l2_{j}.csv
    with cv_name in {'cv_1', 'cv_2', 'cv_3', 'full'} (1-indexed for CVs).
    """
    os.makedirs(output_dir, exist_ok=True)

    original_cols = list(ports.columns)
    new_cols = _rename_tree_cols(original_cols) if is_tree else original_cols

    if is_tree:
        adj_w = compute_adj_w(original_cols)
    else:
        adj_w = np.ones(ports.shape[1])

    # Scale columns (adj_ports in R)
    arr = ports.to_numpy(dtype=float) * adj_w

    T = arr.shape[0]
    n_valid = n_train_valid // cv_n
    test_start = n_train_valid
    ports_test = arr[test_start:T]

    folds = range(1, cv_n + 1) if run_full_cv else [cv_n]

    for i_fold in folds:
        valid_start = (i_fold - 1) * n_valid
        valid_end = i_fold * n_valid
        valid_idx = np.arange(valid_start, valid_end)
        train_mask = np.ones(T, dtype=bool)
        train_mask[valid_idx] = False
        train_mask[test_start:] = False
        ports_train = arr[train_mask]
        ports_valid = arr[valid_idx]
        cv_name = f"cv_{i_fold}"
        _grid_search(
            ports_train, ports_valid, ports_test,
            lambda0_list, lambda2_list, adj_w, new_cols,
            output_dir, cv_name, kmin, kmax, n_workers=n_workers,
        )

    # Final full fit on [0:n_train_valid]
    ports_train_full = arr[:n_train_valid]
    _grid_search(
        ports_train_full, None, ports_test,
        lambda0_list, lambda2_list, adj_w, new_cols,
        output_dir, "full", kmin, kmax, n_workers=n_workers,
    )


def _grid_search(
    ports_train, ports_valid, ports_test,
    lambda0_list, lambda2_list, adj_w, col_names,
    output_dir, cv_name, kmin, kmax, n_workers=1,
):
    # Populate module-level context so fork-based workers inherit arrays
    # without re-pickling them per task.
    _WORKER_CTX.clear()
    _WORKER_CTX.update({
        "ports_train": ports_train,
        "ports_valid": ports_valid,
        "ports_test":  ports_test,
        "adj_w":       adj_w,
        "col_names":   col_names,
        "kmin":        kmin,
        "kmax":        kmax,
        "output_dir":  output_dir,
        "cv_name":     cv_name,
    })

    tasks = [
        (i, j, l0, l2)
        for i, l0 in enumerate(lambda0_list, start=1)
        for j, l2 in enumerate(lambda2_list, start=1)
    ]

    t0 = time.time()
    print(
        f"[ap_pruning] fold={cv_name} tasks={len(tasks)} n_workers={n_workers} ...",
        flush=True,
    )
    if n_workers <= 1:
        for t in tasks:
            _process_config(t)
    else:
        with Pool(processes=n_workers) as pool:
            # Use imap_unordered for progress visibility; workers fork here.
            for _ in pool.imap_unordered(_process_config, tasks, chunksize=1):
                pass
    print(
        f"[ap_pruning] fold={cv_name} done in {time.time() - t0:.1f}s",
        flush=True,
    )
