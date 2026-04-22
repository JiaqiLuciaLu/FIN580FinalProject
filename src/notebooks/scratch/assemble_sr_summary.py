"""Assemble SR_Summary.csv across all 36 cross-sections for Figure 6a.

For each subdir under TreeGridSearch/:
  - AP-Tree K=10: pick test SR at argmax(valid_SR_10).
  - AP-Tree K=40: pick test SR at argmax(valid_SR_40).
  - TS32:         pick test SR at argmax(valid_SR_32)  (TSGridSearch).
  - TS64:         pick test SR at argmax(valid_SR_64)  (TS64GridSearch).

Writes `<OUTPUT_DIR>/tables/SR_Summary.csv` with column layout matching
`reference_code/4_Plots/Figure6a_7_8_SR_Plot_XSF.R` (1-indexed):
    col 4  Id        col 5  ts32      col 6  ts64
    col 8  xsf       col 10 aptree10  col 11 aptree40

XSF column is a NaN placeholder (we have no XSF data).

Then calls `sr_plot_xsf` to render `SRwithXSF_{10,40}.png` without the XSF curve.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.code import utils
from src.code.plots.figure6a_sr_plot_xsf import sr_plot_xsf


def _best_test_sr(grid_root, subdir, K):
    """Return test SR at argmax(valid_SR_K) for a single (grid_root, subdir)."""
    valid_path = os.path.join(grid_root, subdir, f"valid_SR_{K}.csv")
    test_path = os.path.join(grid_root, subdir, f"test_SR_{K}.csv")
    if not (os.path.exists(valid_path) and os.path.exists(test_path)):
        return np.nan
    valid = pd.read_csv(valid_path).to_numpy()
    test = pd.read_csv(test_path).to_numpy()
    if np.all(np.isnan(valid)):
        return np.nan
    flat = int(np.nanargmax(valid))
    i, j = np.unravel_index(flat, valid.shape)
    return float(test[i, j])


def list_subdirs():
    base = Path(utils.PY_TREE_GRID_DIR)
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and d.name.startswith("LME_")
    )


def assemble(out_csv):
    rows = []
    for sub in list_subdirs():
        ap10 = _best_test_sr(utils.PY_TREE_GRID_DIR, sub, 10)
        ap40 = _best_test_sr(utils.PY_TREE_GRID_DIR, sub, 40)
        ts32 = _best_test_sr(utils.PY_TS_GRID_DIR, sub, 32)
        ts64 = _best_test_sr(utils.PY_TS64_GRID_DIR, sub, 64)

        # Skip subdirs missing any required SR (e.g. LME_r12_2_IdioVol TS32 failure).
        if np.isnan([ap10, ap40, ts32, ts64]).any():
            print(f"[assemble] skip {sub}: "
                  f"ap10={ap10}, ap40={ap40}, ts32={ts32}, ts64={ts64}")
            continue

        # R 1-indexed columns: 4=Id, 5=ts32, 6=ts64, 8=xsf, 10=ap10, 11=ap40.
        # Fill the rest with NaN placeholders to keep column indexing intact.
        rows.append({
            "c1": np.nan, "c2": np.nan, "c3": np.nan,
            "Id": sub,
            "ts32": ts32,
            "ts64": ts64,
            "c7": np.nan,
            "xsf": np.nan,
            "c9": np.nan,
            "aptree10": ap10,
            "aptree40": ap40,
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[assemble] wrote {out_csv} ({len(df)} rows)")
    return out_csv


def main():
    out_csv = os.path.join(utils.OUTPUT_DIR, "tables", "SR_Summary.csv")
    assemble(out_csv)

    plot_dir = os.path.join(utils.PROJECT_ROOT, "plots", "Figure6")
    for p in (10, 40):
        path = sr_plot_xsf(out_csv, plot_dir, p=p)
        print(f"[assemble] wrote {path}")


if __name__ == "__main__":
    main()
