"""Python translation of `reference_code/2_AP_Pruning/AP_Pruning.R`.

Read the filtered tree portfolios, compute per-column depth-adjustment
weights, rescale the ports accordingly, and dispatch to `lasso_valid_full`.
"""

import os

import numpy as np
import pandas as pd

from src.code import utils
from src.code.ap_pruning.lasso_valid_par_full import lasso_valid_full


def AP_Pruning(feats_list=None, feat1=utils.FEAT1, feat2=utils.FEAT2,
               input_path=utils.PY_TREE_PORT_DIR,
               input_file_name="level_all_excess_combined_filtered.csv",
               output_path=utils.PY_TREE_GRID_DIR,
               n_train_valid=utils.N_TRAIN_VALID, cvN=utils.CV_N,
               runFullCV=False, kmax=utils.KMAX,
               RunParallel=False, ParallelN=10,
               IsTree=True, lambda0=None, lambda2=None):
    if feats_list is None:
        feats_list = utils.FEATS_LIST

    chars = ["LME", feats_list[feat1 - 1], feats_list[feat2 - 1]]
    subdir = "_".join(chars)
    print(subdir)

    ports = pd.read_csv(os.path.join(input_path, subdir, input_file_name))

    if IsTree:
        # R reads CSVs with X-prefixed column names (R: read.csv auto-adds X
        # to names starting with a digit). Python pandas does NOT add the X.
        # Simulate R's X-prefixing so the substring arithmetic below matches.
        coln_x = ["X" + c for c in ports.columns]
        depths = np.array([len(c) - 7 for c in coln_x])

        coln = list(coln_x)
        for i in range(len(coln)):
            # R: paste(substring(coln[i], 2, nchar(substring(coln[i], 7, 11))),
            #         substring(coln[i], 7, 11), sep='.')
            path_part = coln[i][6:11]          # R chars 7..11 -> Py 0-indexed [6:11]
            path_len = len(path_part)
            perm_part = coln[i][1:path_len]     # R chars 2..path_len -> Py [1:path_len]
            coln[i] = f"{perm_part}.{path_part}"
        ports.columns = coln
    else:
        depths = np.zeros(ports.shape[1], dtype=int)

    adj_w = 1 / np.sqrt(2.0 ** depths)

    adj_ports = ports.copy()
    if IsTree:
        for i in range(len(adj_w)):
            adj_ports.iloc[:, i] = ports.iloc[:, i].values * adj_w[i]

    lasso_valid_full(
        adj_ports, lambda0, lambda2, output_path, subdir, adj_w,
        n_train_valid=n_train_valid, cvN=cvN, runFullCV=runFullCV,
        kmax=kmax, RunParallel=RunParallel, ParallelN=ParallelN,
    )
