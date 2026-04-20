"""Python translation of
`reference_code/1_Portfolio_Creation/Tree_Portfolio_Creation/Step1_Combine_Raw_Chars_Convert_Quantile_Split_Yearly_Chunks.R`.
"""

import os

import numpy as np
import pandas as pd

from src.code import utils


def convert_quantile(x):
    """R: x[!is.na(x)] = (rank(na.omit(x)) - 1) / (length(na.omit(x)) - 1)."""
    x = np.asarray(x, dtype=float).copy()
    not_na = ~np.isnan(x)
    n = int(not_na.sum())
    if n <= 1:
        # R: division by zero when n == 1 -> NaN. n == 0 -> leave empty.
        if n == 1:
            x[not_na] = np.nan
        return x
    ranks = pd.Series(x[not_na]).rank(method="average").values
    x[not_na] = (ranks - 1) / (n - 1)
    return x


def create_yearly_chunks(y_min=utils.Y_MIN, y_max=utils.Y_MAX,
                         feats_list=None, feat1=utils.FEAT1, feat2=utils.FEAT2,
                         input_path=utils.CHAR_PANEL_DIR,
                         output_path=utils.PY_DATA_CHUNK_DIR,
                         add_noise=False):
    if feats_list is None:
        feats_list = utils.FEATS_LIST
    print(feat1)
    print(feat2)
    feats = ["LME", feats_list[feat1 - 1], feats_list[feat2 - 1]]

    os.makedirs(os.path.join(output_path, "_".join(feats)), exist_ok=True)

    RET = pd.read_csv(os.path.join(input_path, "RET.csv"))

    features = []
    for i in range(len(feats)):
        features.append(pd.read_csv(os.path.join(input_path, f"{feats[i]}.csv")))

    date = RET["date"].values
    RET = RET.iloc[:, 1:]
    RET.columns = [c[4:] for c in RET.columns]  # strip 'RET.' (4 chars)

    for i in range(len(feats)):
        features[i] = features[i].iloc[:, 1:]
        prefix_len = len(feats[i]) + 1  # '<feat>.'
        features[i].columns = [c[prefix_len:] for c in features[i].columns]

    # Convert raw characteristics to Quantile numbers.
    # R iterates row-by-row (per-month) and writes back via features[[i]][j,]=...
    # In pandas, .iloc[j, :] = arr on a ~20k-column DataFrame is pathologically
    # slow (~minutes per row). Replace with panel-wise rank along axis=1 — same
    # per-month ranking, one vectorized call instead of 636 row-writes.
    # Numerically identical: R's rank(na.omit(x)) uses ties.method="average",
    # which matches pandas .rank(method="average"); NA rows stay NA via mask.
    nrow_feat = features[0].shape[0]
    print(f"Quantile-normalizing {len(feats)} panels of {nrow_feat} months each...")
    for i in range(len(feats)):
        arr = features[i].to_numpy(dtype=float)
        nan_mask = np.isnan(arr)
        ranks = pd.DataFrame(arr).rank(axis=1, method="average").to_numpy()
        counts = (~nan_mask).sum(axis=1, keepdims=True).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            normalized = (ranks - 1.0) / np.maximum(counts - 1.0, 1.0)
        normalized[nan_mask] = np.nan
        features[i] = pd.DataFrame(
            normalized, index=features[i].index, columns=features[i].columns
        )
        print(f"  feat {i + 1}/{len(feats)} ({feats[i]}) normalized")

    # Add the raw size info for weighting purpose.
    # R reads lowercase 'lme.csv'; our CHAR_PANEL_DIR ships uppercase 'LME.csv'
    # (same underlying panel). Column-prefix length (4 chars) is identical.
    lme = pd.read_csv(os.path.join(input_path, "LME.csv"))
    lme = lme.iloc[:, 1:]
    lme.columns = [c[4:] for c in lme.columns]  # strip 'LME.' (4 chars)

    # Combine the info and save the data into yearly files
    for year in range(y_min, y_max + 1):
        rows = []

        for month in range(1, 13):
            i = 12 * (year - y_min) + month - 1
            retm = RET.iloc[i, :].dropna()
            feature_m = [features[j].iloc[i, :].dropna() for j in range(len(feats))]
            lmem = lme.iloc[i, :].dropna()

            # R's intersect preserves order of first argument
            set0 = set(feature_m[0].index)
            inter = [c for c in retm.index if c in set0]
            for j in range(1, len(feats)):
                set_j = set(feature_m[j].index)
                inter = [c for c in inter if c in set_j]
            set_lme = set(lmem.index)
            inter = [c for c in inter if c in set_lme]

            if len(inter) == 0:
                continue

            data_m = pd.DataFrame({
                "yy": [year] * len(inter),
                "mm": [month] * len(inter),
                "date": [date[i]] * len(inter),
                "permno": inter,
                "ret": retm.loc[inter].astype(float).values,
            })
            for j in range(len(feats)):
                data_m[feats[j]] = feature_m[j].loc[inter].astype(float).values
            data_m["size"] = lmem.loc[inter].astype(float).values

            rows.append(data_m)

        data_train = (
            pd.concat(rows, ignore_index=True)
            if rows
            else pd.DataFrame(columns=["yy", "mm", "date", "permno", "ret", *feats, "size"])
        )

        if add_noise:
            n = len(data_train)
            data_train["ret"] = np.maximum(
                data_train["ret"].values + np.random.normal(0, 0.05, n), -0.99
            )
            data_train["size"] = np.maximum(
                data_train["size"].values + np.random.normal(0, 0.05, n), 0.01
            )

        out_path = os.path.join(output_path, "_".join(feats), f"y{year}.csv")
        data_train.to_csv(out_path, index=False)
