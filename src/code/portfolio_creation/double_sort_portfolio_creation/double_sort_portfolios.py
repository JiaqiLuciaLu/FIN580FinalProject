"""Python translation of
`reference_code/1_Portfolio_Creation/Double_Sort_Portfolio_Creation/DoubleSort Portfolios.R`.

Build double-sorted portfolios: q_num x q_num grid of value-weighted
returns per (feat1, feat2) pair, subtract the risk-free rate, write CSV.
"""

import os

import numpy as np
import pandas as pd

from src.code import utils
from src.code.dplyr_shim import ntile


def double_sort_helper(df_tmp, feat1, feat2, q_num):
    """Build portfolio for a specific year.

    Input: df_tmp — dataframe of a specific year
           feat1, feat2 — two features
           q_num — number of cuts (usually 2 or 5)
    Output: ret_tmp — portfolio returns over a specific year, shape (q_num^2, 12)
    """
    ret_tmp = np.zeros((q_num * q_num, 12))
    df_tmp["1"] = 0
    df_tmp["2"] = 0
    for mon in range(1, 13):
        mask_m = df_tmp["mm"] == mon
        df_m = df_tmp.loc[mask_m, :].copy()
        df_m["1"] = ntile(df_m[feat1].values, q_num)
        df_m["2"] = ntile(df_m[feat2].values, q_num)
        for i in range(1, q_num + 1):
            for j in range(1, q_num + 1):
                mask_port = (df_m["1"] == i) & (df_m["2"] == j)
                company_val = df_m.loc[mask_port, "size"].values
                ret_mon = df_m.loc[mask_port, "ret"].values
                total = np.sum(company_val)
                if total != 0:
                    ret_tmp[(i - 1) * q_num + j - 1, mon - 1] = (
                        np.dot(ret_mon, company_val) / total
                    )
                else:
                    ret_tmp[(i - 1) * q_num + j - 1, mon - 1] = np.nan
    return ret_tmp


def double_sort(data_path, feat1, feat2, q_num, y_min, y_max):
    """Build portfolios and compute the value-averaged returns.

    Input: data_path — path for the data
           feat1, feat2 — two features
           q_num — number of cuts (usually 2 or 5)
           y_min, y_max — the range of years
    """
    ret_table = np.zeros((q_num * q_num, (y_max - y_min + 1) * 12))
    y_time_stamp = 1
    for y in range(y_min, y_max + 1):
        print(y)
        data_filenm = os.path.join(data_path, f"y{y}.csv")
        df_tmp = pd.read_csv(data_filenm)
        ret_table[:, (y_time_stamp - 1) * 12:y_time_stamp * 12] = double_sort_helper(
            df_tmp, feat1, feat2, q_num
        )
        y_time_stamp += 1
    return ret_table


def remove_rf(port_ret, factor_path):
    file_nm = os.path.join(factor_path, "rf_factor.csv")
    r_f = pd.read_csv(file_nm, header=None).iloc[:, 0].values.astype(float)
    for i in range(port_ret.shape[1]):
        port_ret[:, i] = port_ret[:, i] - r_f / 100
    return port_ret


###################
### Main code   ###
###################


def main(feats_list=None, q_num=4, y_min=utils.Y_MIN, y_max=utils.Y_MAX,
         data_chunk_path=utils.DATA_CHUNK_DIR,
         output_path=utils.PY_DS_PORT_DIR,
         factor_path=utils.FACTOR_DIR):
    if feats_list is None:
        feats_list = utils.FEATS_LIST

    for feat1n in range(1, len(feats_list)):  # R: 1:(length-1)
        for feat2n in range(feat1n + 1, len(feats_list) + 1):  # R: (feat1n+1):length
            print(feat1n)
            print(feat2n)

            feat1 = feats_list[feat1n - 1]
            feat2 = feats_list[feat2n - 1]

            sub_dir = f"{feat1}_{feat2}"
            os.makedirs(os.path.join(output_path, sub_dir), exist_ok=True)
            data_path = os.path.join(data_chunk_path, sub_dir) + "/"

            ret_table = double_sort(data_path, feat1, feat2, q_num, y_min, y_max)
            print(np.sum(np.isnan(ret_table)))

            ret_table = ret_table.T  # t() in R

            port_ret = remove_rf(ret_table, factor_path)
            port_ret[np.isnan(port_ret)] = 0

            out_file = os.path.join(
                output_path, f"{feat1}_{feat2}", f"ds_{q_num ** 2}excess.csv"
            )
            pd.DataFrame(port_ret).to_csv(out_file, index=False)


if __name__ == "__main__":
    main()
