"""Python translation of
`reference_code/1_Portfolio_Creation/Triple_Sort_Portfolio_Creation/TripleSort32_Portfolios.R`.

Build 2x4x4 = 32 triple-sorted portfolios for (LME, feat1, feat2), subtract
risk-free, write excess_ports.csv.
"""

import os

import numpy as np
import pandas as pd

from src.code import utils
from src.code.dplyr_shim import ntile


def triple_sort_helper(df_tmp, feat1, feat2, feat3):
    ret_tmp = np.zeros((32, 12))
    df_tmp["1"] = 0
    df_tmp["2"] = 0
    df_tmp["3"] = 0
    for mon in range(1, 13):
        mask_m = df_tmp["mm"] == mon
        df_m = df_tmp.loc[mask_m, :].copy()
        df_m["1"] = ntile(df_m[feat1].values, 2)
        df_m["2"] = ntile(df_m[feat2].values, 4)
        df_m["3"] = ntile(df_m[feat3].values, 4)
        for i in range(1, 3):
            for j in range(1, 5):
                for k in range(1, 5):
                    mask_port = (df_m["1"] == i) & (df_m["2"] == j) & (df_m["3"] == k)
                    company_val = df_m.loc[mask_port, "size"].values
                    ret_mon = df_m.loc[mask_port, "ret"].values
                    total = np.sum(company_val)
                    row = (i - 1) * 16 + (j - 1) * 4 + k - 1
                    if total != 0:
                        ret_tmp[row, mon - 1] = np.dot(ret_mon, company_val) / total
                    else:
                        ret_tmp[row, mon - 1] = np.nan
    return ret_tmp


def triple_sort(data_path, feat1, feat2, feat3, y_min, y_max):
    ret_table = np.zeros((32, (y_max - y_min + 1) * 12))
    y_time_stamp = 1
    for y in range(y_min, y_max + 1):
        print(y)
        data_filenm = os.path.join(data_path, f"y{y}.csv")
        df_tmp = pd.read_csv(data_filenm)
        ret_table[:, (y_time_stamp - 1) * 12:y_time_stamp * 12] = triple_sort_helper(
            df_tmp, feat1, feat2, feat3
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


def gen_triple_sort_32(feats_list, feat1, feat2,
                       y_min=utils.Y_MIN, y_max=utils.Y_MAX,
                       data_chunk_path=utils.DATA_CHUNK_DIR,
                       output_path=utils.PY_TS_PORT_DIR,
                       factor_path=utils.FACTOR_DIR):
    print(feat1)
    print(feat2)
    feats = ["LME", feats_list[feat1 - 1], feats_list[feat2 - 1]]

    sub_dir = f"{feats[0]}_{feats[1]}_{feats[2]}"

    os.makedirs(os.path.join(output_path, sub_dir), exist_ok=True)
    data_path = os.path.join(data_chunk_path, sub_dir) + "/"

    ret_table = triple_sort(data_path, feats[0], feats[1], feats[2], y_min, y_max)
    print(np.sum(np.isnan(ret_table)))

    ret_table = ret_table.T

    port_ret = remove_rf(ret_table, factor_path)
    port_ret[np.isnan(port_ret)] = 0

    out_file = os.path.join(output_path, sub_dir, "excess_ports.csv")
    pd.DataFrame(port_ret).to_csv(out_file, index=False)
