"""Python translation of
`reference_code/1_Portfolio_Creation/Tree_Portfolio_Creation/tree_portfolio_helper.R`.

Generate value-weighted returns and min/max of characteristics of tree
portfolios for one tree.
"""

import numpy as np
import pandas as pd

from src.code.dplyr_shim import ntile


def tree_portfolio_y_helper(df_m_recurse, feat_list, k, tree_depth, q_num):
    k = k + 1
    df_m_recurse[str(k)] = ntile(df_m_recurse[feat_list[k - 1]].values, q_num)
    if k < tree_depth:
        for val in range(1, q_num + 1):
            mask_recurse = df_m_recurse[str(k)] == val
            df_m_recurse.loc[mask_recurse, :] = tree_portfolio_y_helper(
                df_m_recurse.loc[mask_recurse, :].copy(), feat_list, k, tree_depth, q_num
            )
    return df_m_recurse


def tree_portfolio_y(df_tmp, feat_list, tree_depth, q_num):
    for k in range(1, tree_depth + 1):
        df_tmp[str(k)] = 0
    for i in range(0, tree_depth + 1):
        df_tmp[f"port{i}"] = 1

    for m in range(1, 13):
        mask_m = df_tmp["mm"] == m
        df_m = df_tmp.loc[mask_m, :].copy()
        df_m[str(1)] = ntile(df_m[feat_list[0]].values, q_num)
        for val in range(1, q_num + 1):
            k = 1
            mask_m_tmp = df_m[str(1)] == val
            df_m_recurse = df_m.loc[mask_m_tmp, :].copy()
            df_m.loc[mask_m_tmp, :] = tree_portfolio_y_helper(
                df_m_recurse, feat_list, k, tree_depth, q_num
            )
        df_tmp.loc[mask_m, :] = df_m
    return df_tmp


def tree_portfolio(data_path, feat_list, tree_depth, q_num, y_min, y_max, file_prefix, feats):
    n_feats = len(feats)
    ret_table = np.zeros(((y_max - y_min + 1) * 12, q_num ** (tree_depth + 1) - 1))
    feat_min_table = [
        np.zeros(((y_max - y_min + 1) * 12, q_num ** (tree_depth + 1) - 1))
        for _ in range(n_feats)
    ]
    feat_max_table = [
        np.zeros(((y_max - y_min + 1) * 12, q_num ** (tree_depth + 1) - 1))
        for _ in range(n_feats)
    ]

    for y in range(y_min, y_max + 1):
        if y % 5 == 0:
            print(y)
        data_filenm = f"{data_path}{file_prefix}{y}.csv"
        df_m = pd.read_csv(data_filenm)
        df_m = tree_portfolio_y(df_m, feat_list, tree_depth, q_num)

        for i in range(1, tree_depth + 1):
            for k in range(1, i + 1):
                df_m[f"port{i}"] = df_m[f"port{i}"] + (df_m[str(k)] - 1) * (q_num ** (i - k))

        for i in range(0, tree_depth + 1):
            for m in range(1, 13):
                for k in range(1, q_num ** i + 1):
                    mask_port = (df_m["mm"] == m) & (df_m[f"port{i}"] == k)
                    company_val = df_m.loc[mask_port, "size"].values
                    ret_mon = df_m.loc[mask_port, "ret"].values

                    row = 12 * (y - y_min) + m - 1
                    col = 2 ** i - 1 + k - 1

                    total = np.sum(company_val)
                    if total != 0:
                        ret_table[row, col] = np.dot(ret_mon, company_val) / total
                    else:
                        ret_table[row, col] = np.nan

                    for f in range(n_feats):
                        feat_vals = df_m.loc[mask_port, feats[f]].values
                        if len(feat_vals) > 0:
                            feat_min_table[f][row, col] = np.min(feat_vals)
                            feat_max_table[f][row, col] = np.max(feat_vals)
                        else:
                            feat_min_table[f][row, col] = np.inf
                            feat_max_table[f][row, col] = -np.inf

    ret_list = [ret_table]
    for f in range(n_feats):
        ret_list.append(feat_min_table[f])
        ret_list.append(feat_max_table[f])
    return ret_list
