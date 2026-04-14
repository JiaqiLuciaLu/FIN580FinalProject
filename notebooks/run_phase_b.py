"""
Phase B: run ap_pruning for the single-config case matching main_simplified.R,
then cross-validate against R's ground-truth CSV.

Config:
  λ₀ = 0.15  (idx 4 in R full grid seq(0, 0.9, 0.05))
  λ₂ = 1e-8  (idx 13 in R full grid 0.1^seq(5, 8, 0.25))
  runFullCV = False  => only fold 3
  kmin = 5, kmax = 50
"""

import os
import time
import numpy as np
import pandas as pd

from src import data_prep, pruning, utils


def main():
    print("=" * 70)
    print("Phase B: single-config AP pruning")
    print("=" * 70)

    ports = data_prep.load_filtered_tree_portfolios()
    print(f"Loaded filtered tree portfolios: shape={ports.shape}")

    out_dir = os.path.join(utils.PY_TREE_GRID_DIR, utils.SUBDIR_3CHAR)
    os.makedirs(out_dir, exist_ok=True)

    lambda0_list = [0.15]
    lambda2_list = [1e-8]

    t0 = time.time()
    pruning.ap_pruning(
        ports,
        lambda0_list=lambda0_list,
        lambda2_list=lambda2_list,
        output_dir=out_dir,
        n_train_valid=utils.N_TRAIN_VALID,
        cv_n=utils.CV_N,
        run_full_cv=False,
        kmin=utils.KMIN,
        kmax=utils.KMAX,
        is_tree=True,
    )
    elapsed = time.time() - t0
    print(f"AP pruning finished in {elapsed:.1f}s")
    print(f"Wrote outputs to: {out_dir}")
    for f in sorted(os.listdir(out_dir)):
        size = os.path.getsize(os.path.join(out_dir, f))
        print(f"  {size:>10,} bytes  {f}")

    # Cross-validate against R ground truth
    print()
    print("=" * 70)
    print("Cross-validation vs R ground truth")
    print("=" * 70)
    r_dir = os.path.join(utils.TREE_GRID_DIR, utils.SUBDIR_3CHAR)
    # R full-grid indices for our single config: l0 = 0.15 -> index 4, l2 = 1e-8 -> index 13
    pairs = [
        ("results_cv_3_l0_1_l2_1.csv", "results_cv_3_l0_4_l2_13.csv"),
        ("results_full_l0_1_l2_1.csv", "results_full_l0_4_l2_13.csv"),
    ]
    for py_name, r_name in pairs:
        py_path = os.path.join(out_dir, py_name)
        r_path = os.path.join(r_dir, r_name)
        compare(py_path, r_path)


def compare(py_path, r_path):
    py_df = pd.read_csv(py_path)
    r_df = pd.read_csv(r_path)
    print(f"\n[compare] {os.path.basename(py_path)}  vs  {os.path.basename(r_path)}")
    print(f"  Python shape: {py_df.shape}   R shape: {r_df.shape}")
    print(f"  Python cols (first 5): {py_df.columns[:5].tolist()}")
    print(f"  R      cols (first 5): {r_df.columns[:5].tolist()}")

    # Match on the meta columns that both have
    meta = ["train_SR", "test_SR", "portsN"]
    if "valid_SR" in py_df.columns and "valid_SR" in r_df.columns:
        meta.insert(1, "valid_SR")
    py_meta = py_df[meta]
    r_meta = r_df[meta]
    n = min(len(py_meta), len(r_meta))

    print(f"  First 6 rows side-by-side (Python | R):")
    for i in range(min(6, n)):
        py_row = py_meta.iloc[i].to_dict()
        r_row = r_meta.iloc[i].to_dict()
        print(f"    row {i}: PY={fmt(py_row)}  R={fmt(r_row)}")

    # Numerical comparison on SRs for matched-length prefix
    for col in meta:
        if col == "portsN":
            match = (py_meta[col].iloc[:n].values == r_meta[col].iloc[:n].values).sum()
            print(f"  portsN exact match: {match}/{n}")
        else:
            diff = np.abs(py_meta[col].iloc[:n].values - r_meta[col].iloc[:n].values)
            print(f"  {col}: max|Δ|={diff.max():.2e}  mean|Δ|={diff.mean():.2e}")


def fmt(d):
    return "{" + ", ".join(
        f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items()
    ) + "}"


if __name__ == "__main__":
    main()
