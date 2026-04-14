"""
Data loaders for the AP-Trees replication.

Mirrors the file reads done in `reference_code/main_simplified.R` and
`reference_code/2_AP_Pruning/AP_Pruning.R`. Returns pandas DataFrames
with the same orientation (rows = months, cols = portfolios / factors).
"""

import os
import numpy as np
import pandas as pd

from . import utils


def load_factors():
    """Load `tradable_factors.csv` (636 months × 13 columns)."""
    path = os.path.join(utils.FACTOR_DIR, "tradable_factors.csv")
    df = pd.read_csv(path)
    return df


def load_rf():
    """
    Load `rf_factor.csv` as a 1D numpy array of monthly risk-free rates (%).
    The raw file has no header and a single numeric column.
    """
    path = os.path.join(utils.FACTOR_DIR, "rf_factor.csv")
    df = pd.read_csv(path, header=None)
    return df.iloc[:, 0].to_numpy(dtype=float)


def load_filtered_tree_portfolios(subdir=utils.SUBDIR_3CHAR):
    """
    Load `level_all_excess_combined_filtered.csv` — base portfolios for AP pruning.
    Shape: (n_months=632, n_portfolios=~2200).
    Column names encode tree id + path; preserved as-is so we can parse depths.
    """
    path = os.path.join(
        utils.TREE_PORT_DIR, subdir, "level_all_excess_combined_filtered.csv"
    )
    df = pd.read_csv(path)
    return df


def extract_depths(columns):
    """
    Mirror of R's `depths = nchar(colnames(ports)) - 7` in AP_Pruning.R:5–32.

    R column names look like 'X1111.111X' where the last 5 characters after the
    dot encode the tree path (5 chars -> depth 4 for the leaf, shorter for
    ancestors). `nchar - 7` gives depth.

    We replicate that formula directly on the string length.
    """
    return np.array([len(c) - 7 for c in columns], dtype=int)
