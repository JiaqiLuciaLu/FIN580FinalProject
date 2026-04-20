"""
Sigmoid-split extension — POC on LME_OP_Investment.

Supports two designs (via --method):

  s1 : tensor-product soft tree with sigmoid centered at global 0.5.
       At k→∞ reduces to a hard global-median tensor tree — NOT paper's
       subset-median tree. Produces degenerate universe (only ~28 distinct
       leaves in the hard limit); kept for reference.

  s2 : recursive soft tree with sigmoid centered at the subset-weighted
       median at every node. At k→∞ reduces exactly to paper's subset-median
       hard tree. This is the apples-to-apples extension: any Δ vs the paper
       baseline isolates the effect of softness.

Reports universe-size, pairwise correlation, and leaf return moments vs the
paper-style baseline already on disk. Writes filtered CSV for the chosen
extension design to `data/processed/sigmoid_poc/...`.

Run via slurm/sigmoid_poc.sbatch.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from src.code import utils
from src.code.portfolio_creation import characteristics, tree_construction
from src.code.portfolio_creation.splits import split_sigmoid, split_median_soft

FEATS = ["LME", "OP", "Investment"]
SUBDIR = "LME_OP_Investment"
POC_ROOT = os.path.join(utils.OUTPUT_DIR, "sigmoid_poc")


def _pairwise_corr_stats(df, label, sample=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(df.shape[1], size=min(sample, df.shape[1]), replace=False)
    sub = df.iloc[:, idx].to_numpy(float)
    c = np.corrcoef(sub, rowvar=False)
    off = c[np.triu_indices_from(c, k=1)]
    return {
        "label": label,
        "n_cols": df.shape[1],
        "median_abs_rho": float(np.median(np.abs(off))),
        "mean_abs_rho": float(np.mean(np.abs(off))),
        "q95_abs_rho": float(np.quantile(np.abs(off), 0.95)),
    }


def _leaf_stats(df, label):
    arr = df.to_numpy(float)
    return {
        "label": label,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=("s1", "s2"), default="s2",
                    help="s1 = tensor-product (global 0.5 center); "
                         "s2 = recursive, subset-weighted-median centers. Default s2.")
    ap.add_argument("--k", type=float, default=8.0,
                    help="Sigmoid steepness.")
    args = ap.parse_args()

    os.makedirs(POC_ROOT, exist_ok=True)
    sigmoid_dir = os.path.join(POC_ROOT, f"{args.method}_sigmoid_k{int(args.k)}", SUBDIR)

    print(f"[POC] method={args.method}  k={args.k}  outputs -> {POC_ROOT}", flush=True)

    print("[1/3] Loading clean-CRSP chunks...", flush=True)
    t0 = time.time()
    chunks = characteristics.build_chunks_from_raw(feat1=utils.FEAT1, feat2=utils.FEAT2)
    print(f"      {len(chunks)} yearly chunks in {time.time()-t0:.1f}s", flush=True)

    print("[2/3] Loading rf vector...", flush=True)
    rf = tree_construction.load_rf_vector()

    print(f"[3/3] Building sigmoid tree ({args.method.upper()}, k={args.k})...", flush=True)
    t0 = time.time()
    if args.method == "s1":
        sigmoid = tree_construction.build_cross_section(
            chunks, FEATS, rf, utils.Y_MIN, utils.Y_MAX,
            soft_split_fn=split_sigmoid(k=args.k),
            output_dir=sigmoid_dir,
        )
    else:  # s2
        sigmoid = tree_construction.build_cross_section(
            chunks, FEATS, rf, utils.Y_MIN, utils.Y_MAX,
            sigmoid_k_recursive=args.k,
            output_dir=sigmoid_dir,
        )
    print(f"      shape={sigmoid.shape}  elapsed={time.time()-t0:.1f}s", flush=True)

    # Paper-style hard tree already on disk (subset-median baseline)
    paper_like_path = os.path.join(utils.PY_TREE_PORT_DIR, SUBDIR,
                                   "level_all_excess_combined_filtered.csv")
    paper_like = pd.read_csv(paper_like_path)

    print()
    print("=== Universe-size comparison ===", flush=True)
    print(f"  Paper-style (subset-median, hard):  {paper_like.shape[1]:>5d} cols", flush=True)
    print(f"  Sigmoid {args.method.upper()} k={args.k}:                 {sigmoid.shape[1]:>5d} cols", flush=True)

    print()
    print("=== Pairwise column correlation (random 300-col sample) ===", flush=True)
    for df, label in [
        (paper_like, "Paper-style subset-median tree"),
        (sigmoid, f"Sigmoid {args.method.upper()} k={args.k}"),
    ]:
        s = _pairwise_corr_stats(df, label)
        print(f"  {s['label']:40s}  n={s['n_cols']:>4d}  "
              f"median |ρ|={s['median_abs_rho']:.3f}  "
              f"mean |ρ|={s['mean_abs_rho']:.3f}  "
              f"q95 |ρ|={s['q95_abs_rho']:.3f}", flush=True)

    print()
    print("=== Leaf return distribution ===", flush=True)
    for df, label in [
        (paper_like, "Paper-style subset-median tree"),
        (sigmoid, f"Sigmoid {args.method.upper()} k={args.k}"),
    ]:
        s = _leaf_stats(df, label)
        print(f"  {s['label']:40s}  "
              f"mean={s['mean']:+.5f}  std={s['std']:.5f}  "
              f"range=[{s['min']:+.3f},{s['max']:+.3f}]", flush=True)

    print(f"\n[POC] done. Artifact:\n"
          f"  sigmoid CSV: {os.path.join(sigmoid_dir, 'level_all_excess_combined_filtered.csv')}",
          flush=True)


if __name__ == "__main__":
    main()
