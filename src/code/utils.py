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

FACTOR_DIR = os.path.join(DATA_RAW, "factor")
DATA_CHUNK_DIR = os.path.join(DATA_RAW, "data_chunk_files_quantile")
TREE_PORT_DIR = os.path.join(DATA_RAW, "tree_portfolio_quantile")
TREE_GRID_DIR = os.path.join(DATA_RAW, "TreeGridSearch")
TS_GRID_DIR = os.path.join(DATA_RAW, "TSGridSearch")
TS64_GRID_DIR = os.path.join(DATA_RAW, "TS64GridSearch")

# Python outputs (keep separate from R ground truth)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
PY_TREE_GRID_DIR = os.path.join(OUTPUT_DIR, "TreeGridSearch")

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
