"""
Grid-search post-processing: pick best (lambda0, lambda2) on validation SR,
extract selected portfolios and weights at a given sparsity level.

Mirrors `reference_code/3_Metrics_Collection/Pick_Best_Lambda.R`.
"""

import os
import numpy as np
import pandas as pd


def _load_sr_at_K(result_dir, fname, K):
    df = pd.read_csv(os.path.join(result_dir, fname))
    rows = df[df["portsN"] == K]
    if len(rows) == 0:
        return None, None
    first = rows.iloc[0]
    # Column positions in R: col 1 = train_SR, col 2 = valid_SR or test_SR
    # For results_full: [train_SR, test_SR, portsN, betas...]
    # For results_cv_*: [train_SR, valid_SR, test_SR, portsN, betas...]
    return first, df


def pick_best_lambda(
    result_dir,
    portfolio_path,
    portN,
    lambda0_list,
    lambda2_list,
    full_cv=False,
    write=True,
):
    """
    Port of pickBestLambda(...).

    Reads results_full_l0_i_l2_j.csv and results_cv_{k}_l0_i_l2_j.csv
    from `result_dir`, extracts SR at sparsity K = portN, builds the
    (len(lambda0), len(lambda2)) heatmaps, picks argmax-on-valid,
    writes Selected_Ports_K.csv and Selected_Ports_Weights_K.csv.

    Returns (train_SR_at_best, valid_SR_at_best, test_SR_at_best).
    """
    n0, n2 = len(lambda0_list), len(lambda2_list)
    train_SR = np.zeros((n0, n2))
    valid_SR = np.zeros((n0, n2))
    test_SR = np.zeros((n0, n2))

    for i in range(n0):
        for j in range(n2):
            full_row, _ = _load_sr_at_K(
                result_dir, f"results_full_l0_{i+1}_l2_{j+1}.csv", portN
            )
            cv3_row, _ = _load_sr_at_K(
                result_dir, f"results_cv_3_l0_{i+1}_l2_{j+1}.csv", portN
            )
            if full_row is None or cv3_row is None:
                raise RuntimeError(f"No portsN={portN} row for l0_{i+1}_l2_{j+1}")
            train_SR[i, j] = full_row["train_SR"]
            test_SR[i, j] = full_row["test_SR"]
            valid_SR[i, j] = cv3_row["valid_SR"]

            if full_cv:
                v1, _ = _load_sr_at_K(
                    result_dir, f"results_cv_1_l0_{i+1}_l2_{j+1}.csv", portN
                )
                v2, _ = _load_sr_at_K(
                    result_dir, f"results_cv_2_l0_{i+1}_l2_{j+1}.csv", portN
                )
                valid_SR[i, j] = (valid_SR[i, j] + v1["valid_SR"] + v2["valid_SR"]) / 3.0

    i_star, j_star = np.unravel_index(np.argmax(valid_SR), valid_SR.shape)

    # Load betas from best (i*, j*) config at K = portN
    full_row, full_df = _load_sr_at_K(
        result_dir, f"results_full_l0_{i_star+1}_l2_{j_star+1}.csv", portN
    )
    # Drop meta columns: train_SR, test_SR, portsN
    beta_cols = [c for c in full_df.columns if c not in ("train_SR", "test_SR", "portsN")]
    weights = full_row[beta_cols].to_numpy(dtype=float)
    nonzero_idx = np.where(weights != 0)[0]
    weights_nonzero = weights[nonzero_idx]

    # Load base portfolios and select nonzero columns
    base_ports = pd.read_csv(portfolio_path)
    selected_ports = base_ports.iloc[:, nonzero_idx]

    if write:
        pd.DataFrame(train_SR).to_csv(
            os.path.join(result_dir, f"train_SR_{portN}.csv"), index=False
        )
        pd.DataFrame(valid_SR).to_csv(
            os.path.join(result_dir, f"valid_SR_{portN}.csv"), index=False
        )
        pd.DataFrame(test_SR).to_csv(
            os.path.join(result_dir, f"test_SR_{portN}.csv"), index=False
        )
        selected_ports.to_csv(
            os.path.join(result_dir, f"Selected_Ports_{portN}.csv"), index=False
        )
        # R writes as a single column via write.table on a vector
        pd.DataFrame({"x": weights_nonzero}).to_csv(
            os.path.join(result_dir, f"Selected_Ports_Weights_{portN}.csv"), index=False
        )

    return (
        float(train_SR[i_star, j_star]),
        float(valid_SR[i_star, j_star]),
        float(test_SR[i_star, j_star]),
    )


def _load_sr_cache(result_dir, n0, n2, full_cv):
    """
    Pre-load meta columns (portsN + SRs) from every grid CSV needed for SR_N.

    Returns a dict {(cv_name, i, j): DataFrame[portsN, train_SR, valid_SR, test_SR]}.
    Reading each file *once* up front avoids the 46x redundant re-reads that
    pickSRN would otherwise issue (one pick_best_lambda call per K).
    """
    cache = {}
    fold_names = ["cv_1", "cv_2", "cv_3", "full"] if full_cv else ["cv_3", "full"]
    for cv in fold_names:
        meta_cols = ["train_SR", "test_SR", "portsN"]
        if cv != "full":
            meta_cols.insert(1, "valid_SR")
        for i in range(1, n0 + 1):
            for j in range(1, n2 + 1):
                fname = f"results_{cv}_l0_{i}_l2_{j}.csv"
                df = pd.read_csv(os.path.join(result_dir, fname), usecols=meta_cols)
                cache[(cv, i, j)] = df
    return cache


def _first_or_nan(df, K, col):
    rows = df[df["portsN"] == K]
    return float(rows.iloc[0][col]) if len(rows) else np.nan


def _pick_best_from_cache(cache, K, n0, n2, full_cv):
    """Argmax-on-valid lookup at sparsity K using a pre-loaded CSV cache.

    Missing K values (LARS-EN path didn't reach this sparsity) are treated as
    NaN and excluded from argmax, matching R's behaviour where max() on NAs
    returns -Inf / first-index fallback via na.rm semantics.
    """
    train_SR = np.full((n0, n2), np.nan)
    valid_SR = np.full((n0, n2), np.nan)
    test_SR = np.full((n0, n2), np.nan)
    for i in range(n0):
        for j in range(n2):
            full_df = cache[("full", i + 1, j + 1)]
            cv3_df = cache[("cv_3", i + 1, j + 1)]
            train_SR[i, j] = _first_or_nan(full_df, K, "train_SR")
            test_SR[i, j] = _first_or_nan(full_df, K, "test_SR")
            valid_SR[i, j] = _first_or_nan(cv3_df, K, "valid_SR")
            if full_cv:
                v1 = cache[("cv_1", i + 1, j + 1)]
                v2 = cache[("cv_2", i + 1, j + 1)]
                valid_SR[i, j] = (
                    valid_SR[i, j]
                    + _first_or_nan(v1, K, "valid_SR")
                    + _first_or_nan(v2, K, "valid_SR")
                ) / 3.0
    # If every entry is NaN (shouldn't happen with a well-formed grid),
    # fall back to returning NaNs.
    if np.all(np.isnan(valid_SR)):
        return (np.nan, np.nan, np.nan)
    i_star, j_star = np.unravel_index(np.nanargmax(valid_SR), valid_SR.shape)
    return (
        float(train_SR[i_star, j_star]),
        float(valid_SR[i_star, j_star]),
        float(test_SR[i_star, j_star]),
    )


def pick_sr_n(
    result_dir,
    portfolio_path,
    lambda0_list,
    lambda2_list,
    kmin=5,
    kmax=50,
    full_cv=False,
):
    """
    Port of pickSRN(...). Loops pick_best_lambda over K in [kmin, kmax],
    assembles a 3 x (kmax-kmin+1) matrix (rows: train/valid/test SR),
    writes SR_N.csv to result_dir matching R's write.table format, and
    returns the matrix.

    Uses a one-shot CSV cache (read each grid file once) -- ~46x faster than
    naively looping pick_best_lambda per K.
    """
    n0, n2 = len(lambda0_list), len(lambda2_list)
    cache = _load_sr_cache(result_dir, n0, n2, full_cv)
    ks = list(range(kmin, kmax + 1))
    cols = []
    for k in ks:
        tr, va, te = _pick_best_from_cache(cache, k, n0, n2, full_cv)
        cols.append([tr, va, te])
    sr_mat = np.array(cols).T  # 3 x len(ks)

    # Match R's write.table(srN, ..., row.names=F) output exactly:
    #   header: "srN","","","",...  (first is variable name, rest empty -- all quoted)
    #   data:   unquoted numerics
    out_path = os.path.join(result_dir, "SR_N.csv")
    header_fields = ['"srN"'] + ['""'] * (len(ks) - 1)
    with open(out_path, "w") as f:
        f.write(",".join(header_fields) + "\n")
        for row in sr_mat:
            f.write(",".join(repr(float(v)) for v in row) + "\n")
    return sr_mat
