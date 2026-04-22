"""Quick validation of `main_fig6` pipeline outputs across cross-sections.

Checks each cross-section subdir under the Tree / TS32 / TS64 grid roots for
required output files, counts NaNs, and reads the test-period Sharpe ratios
at K=10/40 to eyeball sanity.

Usage:
    python src/notebooks/scratch/validation.py
    python src/notebooks/scratch/validation.py --subdirs LME_BEME_OP LME_BEME_r12_2
    python src/notebooks/scratch/validation.py --compare-to LME_OP_Investment

Exit code 0 = all clean, 1 = at least one subdir failed a check.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.code import utils


REQUIRED = {
    "tree": {
        "grid_root": utils.PY_TREE_GRID_DIR,
        "files": [
            "Selected_Ports_10.csv",
            "Selected_Ports_40.csv",
            "Selected_Ports_Weights_10.csv",
            "Selected_Ports_Weights_40.csv",
            "SR_N.csv",
        ],
    },
    "ts32": {
        "grid_root": utils.PY_TS_GRID_DIR,
        "files": ["Selected_Ports_10.csv", "Selected_Ports_32.csv"],
    },
    "ts64": {
        "grid_root": utils.PY_TS64_GRID_DIR,
        "files": ["Selected_Ports_10.csv", "Selected_Ports_64.csv"],
    },
}

# `SDF_regression` writes to <grid_root>/SDFTests/<subdir>/TimeSeriesAlpha.csv.
SDF_ROOTS = {
    "tree": os.path.join(utils.PY_TREE_GRID_DIR, "SDFTests"),
    "ts32": os.path.join(utils.PY_TS_GRID_DIR, "SDFTests"),
    "ts64": os.path.join(utils.PY_TS64_GRID_DIR, "SDFTests"),
}

# SR_N.csv column layout: col idx i → K = i + 5 (K ∈ 5..50 → 46 cols).
K_TO_COL = {K: K - 5 for K in range(5, 51)}


def _read_nans(path):
    df = pd.read_csv(path)
    return df, int(df.isna().sum().sum())


def check_subdir(subdir, n_test_months=276):
    """Return dict of check results for one cross-section."""
    out = {"subdir": subdir, "missing": [], "nans": 0,
           "sr_test_k10": float("nan"), "sr_test_k40": float("nan")}
    missing = out["missing"]

    # 1. File existence + NaN count
    for grid, cfg in REQUIRED.items():
        for fn in cfg["files"]:
            path = os.path.join(cfg["grid_root"], subdir, fn)
            if not os.path.exists(path):
                missing.append(f"{grid}/{fn}")
                continue
            try:
                _, n = _read_nans(path)
                out["nans"] += n
            except Exception as e:
                missing.append(f"{grid}/{fn} (read err: {e})")

    # 2. SR_N.csv: test row (idx 2), K=10 col / K=40 col
    sr_path = os.path.join(utils.PY_TREE_GRID_DIR, subdir, "SR_N.csv")
    if os.path.exists(sr_path):
        sr = pd.read_csv(sr_path)
        if sr.shape != (3, 46):
            missing.append(f"tree/SR_N.csv (shape {sr.shape} != (3, 46))")
        else:
            out["sr_test_k10"] = float(sr.iloc[2, K_TO_COL[10]])
            out["sr_test_k40"] = float(sr.iloc[2, K_TO_COL[40]])

    # 3. Selected_Ports shapes (test slice = last 276 rows)
    for grid, K in [("tree", 10), ("tree", 40), ("ts32", 32), ("ts64", 64)]:
        path = os.path.join(REQUIRED[grid]["grid_root"], subdir,
                            f"Selected_Ports_{K}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            if df.shape[1] != K:
                missing.append(f"{grid}/Selected_Ports_{K}.csv (cols={df.shape[1]} != {K})")
            if df.shape[0] != 636:
                missing.append(f"{grid}/Selected_Ports_{K}.csv (rows={df.shape[0]} != 636)")

    # 4. SDFTests/TimeSeriesAlpha.csv existence + NaN
    for grid, sdf_root in SDF_ROOTS.items():
        path = os.path.join(sdf_root, subdir, "TimeSeriesAlpha.csv")
        if not os.path.exists(path):
            missing.append(f"{grid}/SDFTests/TimeSeriesAlpha.csv")
            continue
        try:
            _, n = _read_nans(path)
            out["nans"] += n
        except Exception as e:
            missing.append(f"{grid}/SDFTests/TimeSeriesAlpha.csv (read err: {e})")

    return out


def list_available_subdirs():
    base = Path(utils.PY_TREE_GRID_DIR)
    if not base.exists():
        return []
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and d.name.startswith("LME_")
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subdirs", nargs="*", default=None,
                    help="Subdirs to validate (default: auto-discover).")
    ap.add_argument("--compare-to", default=None,
                    help="Reference subdir to print alongside (e.g. LME_OP_Investment).")
    args = ap.parse_args()

    subdirs = args.subdirs or list_available_subdirs()
    if not subdirs:
        print("No subdirs found under TreeGridSearch/.")
        sys.exit(1)

    if args.compare_to and args.compare_to not in subdirs:
        subdirs = [args.compare_to] + list(subdirs)

    print(f"Validating {len(subdirs)} cross-section(s):\n")
    header = f"{'subdir':<32} {'files':<8} {'NaN':<5} {'SR(K=10,test)':<14} {'SR(K=40,test)':<14} status"
    print(header)
    print("-" * len(header))

    failures = []
    for sub in subdirs:
        r = check_subdir(sub)
        n_miss = len(r["missing"])
        files_ok = "OK" if n_miss == 0 else f"miss{n_miss}"
        nan_str = str(r["nans"]) if r["nans"] == 0 else f"!!{r['nans']}"
        ok = n_miss == 0 and r["nans"] == 0 and not np.isnan(r["sr_test_k10"])
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures.append(sub)
        ref_tag = " (ref)" if sub == args.compare_to else ""
        print(f"{sub + ref_tag:<32} {files_ok:<8} {nan_str:<5} "
              f"{r['sr_test_k10']:<14.4f} {r['sr_test_k40']:<14.4f} {status}")
        if r["missing"]:
            for m in r["missing"][:3]:
                print(f"    missing: {m}")
            if len(r["missing"]) > 3:
                print(f"    ... and {len(r['missing']) - 3} more")

    print()
    if failures:
        print(f"FAIL: {len(failures)}/{len(subdirs)} failed: {', '.join(failures)}")
        sys.exit(1)
    print(f"PASS: all {len(subdirs)} clean.")


if __name__ == "__main__":
    main()
