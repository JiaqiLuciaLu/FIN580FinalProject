"""Python translation of `reference_code/3_Metrics_Collection/Pick_Best_Lambda.R`.

Scan the (lambda0, lambda2) grid CSVs for a given K, pick the (i*, j*) with
maximum validation Sharpe, and write out the selected portfolios, weights,
and SR matrices.
"""

import os

import numpy as np
import pandas as pd

from src.code import utils


def pickBestLambda(feats_list=None, feat1=utils.FEAT1, feat2=utils.FEAT2,
                   ap_prune_result_path=utils.PY_TREE_GRID_DIR, portN=10,
                   lambda0=None, lambda2=None,
                   portfolio_path=utils.PY_TREE_PORT_DIR,
                   port_name="level_all_excess_combined_filtered.csv",
                   fullCV=False, writetable=True):
    if feats_list is None:
        feats_list = utils.FEATS_LIST
    print(feat1)
    print(feat2)
    feats = ["LME", feats_list[feat1 - 1], feats_list[feat2 - 1]]

    subdir = "_".join(feats)

    lambda0 = np.asarray(lambda0, dtype=float).ravel()
    lambda2 = np.asarray(lambda2, dtype=float).ravel()

    train_SR = np.zeros((len(lambda0), len(lambda2)))
    valid_SR = np.zeros((len(lambda0), len(lambda2)))
    test_SR = np.zeros((len(lambda0), len(lambda2)))

    for i in range(len(lambda0)):
        for j in range(len(lambda2)):
            full_data = pd.read_csv(
                os.path.join(ap_prune_result_path, subdir,
                             f"results_full_l0_{i + 1}_l2_{j + 1}.csv")
            )
            cv_data = pd.read_csv(
                os.path.join(ap_prune_result_path, subdir,
                             f"results_cv_3_l0_{i + 1}_l2_{j + 1}.csv")
            )

            # R: full_data has cols [train_SR, test_SR, portsN, betas...]
            #    cv_data   has cols [train_SR, valid_SR, test_SR, portsN, betas...]
            full_match = full_data[full_data["portsN"] == portN]
            cv_match = cv_data[cv_data["portsN"] == portN]

            train_SR[i, j] = full_match.iloc[0, 0] if len(full_match) > 0 else np.nan
            # Require both cv AND full to contain this portN, so the argmax over
            # valid_SR never selects a cell whose full_data lacks the row (else
            # downstream weight lookup raises IndexError).
            valid_SR[i, j] = (
                cv_match.iloc[0, 1]
                if len(cv_match) > 0 and len(full_match) > 0
                else np.nan
            )
            test_SR[i, j] = full_match.iloc[0, 1] if len(full_match) > 0 else np.nan

            if fullCV:
                cv1 = pd.read_csv(
                    os.path.join(ap_prune_result_path, subdir,
                                 f"results_cv_1_l0_{i + 1}_l2_{j + 1}.csv")
                )
                cv1_match = cv1[cv1["portsN"] == portN]
                valid_SR[i, j] += cv1_match.iloc[0, 1] if len(cv1_match) > 0 else np.nan
                cv2 = pd.read_csv(
                    os.path.join(ap_prune_result_path, subdir,
                                 f"results_cv_2_l0_{i + 1}_l2_{j + 1}.csv")
                )
                cv2_match = cv2[cv2["portsN"] == portN]
                valid_SR[i, j] += cv2_match.iloc[0, 1] if len(cv2_match) > 0 else np.nan
                valid_SR[i, j] /= 3.0

    # R: which(valid_SR == max(valid_SR), arr.ind = TRUE) -> 1-indexed (row, col).
    # Corner case: for large K, every grid cell's cv fold may lack a row with
    # portsN == K (the LARS path never reached that many selected portfolios).
    # valid_SR is then all-NaN. R's max() returns NaN with a warning and the
    # subsequent which()==max comparison yields integer(0) — which propagates to
    # downstream failure. Python is stricter: np.nanargmax raises. Short-circuit
    # here so pickSRN can collect a NaN column and continue to the next K.
    if np.all(np.isnan(valid_SR)):
        return np.array([np.nan, np.nan, np.nan])

    flat_argmax = int(np.nanargmax(valid_SR))
    idx_row, idx_col = np.unravel_index(flat_argmax, valid_SR.shape)

    full_data = pd.read_csv(
        os.path.join(ap_prune_result_path, subdir,
                     f"results_full_l0_{idx_row + 1}_l2_{idx_col + 1}.csv")
    )
    full_match = full_data[full_data["portsN"] == portN]

    # R: weights = full_data[full_data$portsN == portN, ][1, -(1:3)]
    weights = full_match.iloc[0, 3:]
    nonzero_mask = (weights != 0).values
    weights_out = weights[nonzero_mask].values
    ports_no = np.where(nonzero_mask)[0]

    ports = pd.read_csv(os.path.join(portfolio_path, subdir, port_name))
    selected_ports = ports.iloc[:, ports_no]

    if writetable:
        out_dir = os.path.join(ap_prune_result_path, subdir)
        pd.DataFrame(train_SR).to_csv(os.path.join(out_dir, f"train_SR_{portN}.csv"), index=False)
        pd.DataFrame(valid_SR).to_csv(os.path.join(out_dir, f"valid_SR_{portN}.csv"), index=False)
        pd.DataFrame(test_SR).to_csv(os.path.join(out_dir, f"test_SR_{portN}.csv"), index=False)
        selected_ports.to_csv(os.path.join(out_dir, f"Selected_Ports_{portN}.csv"), index=False)
        # R's write.table of a numeric vector uses column header "x"
        pd.DataFrame({"x": weights_out}).to_csv(
            os.path.join(out_dir, f"Selected_Ports_Weights_{portN}.csv"), index=False
        )

    return np.array([
        train_SR[idx_row, idx_col],
        valid_SR[idx_row, idx_col],
        test_SR[idx_row, idx_col],
    ])
