"""
Shared utilities, paths, and constants for the AP-Trees replication.
Mirrors parameters used in `reference_code/main_simplified.R`.
"""

import os

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Shared data lives on scratch for collaborator access
DATA_ROOT = "/scratch/network/jl6134/COLLAB/FIN580/data"
DATA_RAW = os.path.join(DATA_ROOT, "raw")
DATA_DIR = os.path.join(DATA_RAW, "data")        # authors' R-pipeline inputs/outputs
AP_CHAR_DIR = os.path.join(DATA_RAW, "ap_char")  # CRSP per-characteristic CSVs + benchmark AP-Tree portfolios

FACTOR_DIR = os.path.join(DATA_DIR, "factor")
DATA_CHUNK_DIR = os.path.join(DATA_DIR, "data_chunk_files_quantile")
TREE_PORT_DIR = os.path.join(DATA_DIR, "tree_portfolio_quantile")
TREE_GRID_DIR = os.path.join(DATA_DIR, "TreeGridSearch")
TS_GRID_DIR = os.path.join(DATA_DIR, "TSGridSearch")
TS64_GRID_DIR = os.path.join(DATA_DIR, "TS64GridSearch")

# CRSP raw characteristic panels (firm × month) and benchmark AP-Tree portfolios
CHAR_PANEL_DIR = os.path.join(AP_CHAR_DIR, "characteristics")
AP_TREE_3CHAR_DIR = os.path.join(AP_CHAR_DIR, "AP-Tree_3_characteristics")
AP_TREE_10CHAR_DIR = os.path.join(AP_CHAR_DIR, "AP-Tree_10_characteristics")

# Python outputs (on scratch, alongside raw/ — not in the repo)
OUTPUT_DIR = os.path.join(DATA_ROOT, "processed")
PY_TREE_GRID_DIR = os.path.join(OUTPUT_DIR, "TreeGridSearch")
PY_TREE_PORT_DIR = os.path.join(OUTPUT_DIR, "tree_portfolio_quantile")
PY_FUZZY_TREE_PORT_DIR = os.path.join(OUTPUT_DIR, "fuzzy_tree_portfolio_quantile")
PY_FUZZY_TREE_GRID_DIR = os.path.join(OUTPUT_DIR, "FuzzyTreeGridSearch")
PY_DATA_CHUNK_DIR = os.path.join(OUTPUT_DIR, "data_chunk_files_quantile")
PY_DS_PORT_DIR = os.path.join(OUTPUT_DIR, "ds_portfolio")
PY_QUINTILE_PORT_DIR = os.path.join(OUTPUT_DIR, "quintile_portfolios")
PY_TS_PORT_DIR = os.path.join(OUTPUT_DIR, "ts_portfolio")      # triple-sort 32
PY_TS64_PORT_DIR = os.path.join(OUTPUT_DIR, "ts64_portfolio")  # triple-sort 64
PY_TS_GRID_DIR = os.path.join(OUTPUT_DIR, "TSGridSearch")
PY_TS64_GRID_DIR = os.path.join(OUTPUT_DIR, "TS64GridSearch")
PY_PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# ----------------------------------------------------------------------
# Paper constants (matches main_simplified.R)
# ----------------------------------------------------------------------
FEATS_LIST = [
    "LME", "BEME", "r12_2", "OP", "Investment",
    "ST_Rev", "LT_Rev", "AC", "IdioVol", "LTurnover",
]
FEAT1 = 4  # 1-indexed (R-style): OP
FEAT2 = 5  # 1-indexed (R-style): Investment

Y_MIN = 1964
Y_MAX = 2016
N_TRAIN_VALID = 360
CV_N = 3
KMIN = 5
KMAX = 50

# Characteristic subdirectory name (LME_OP_Investment)
SUBDIR_3CHAR = "_".join(["LME", FEATS_LIST[FEAT1 - 1], FEATS_LIST[FEAT2 - 1]])


def char_subdir(feat1=FEAT1, feat2=FEAT2, feats_list=FEATS_LIST):
    """Return characteristic subdirectory name, e.g. 'LME_OP_Investment'."""
    return "_".join(["LME", feats_list[feat1 - 1], feats_list[feat2 - 1]])
