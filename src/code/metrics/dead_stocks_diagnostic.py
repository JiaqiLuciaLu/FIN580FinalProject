"""Dead-stocks diagnostic for fuzzy AP-Tree leaves.

A 'dead stock' in a fuzzy leaf is a firm whose membership weight `w_i^L`
falls below `threshold` (default 0.1%). At each tree depth, fuzzy
membership is real-valued in (0, parent_w), so 'dead' is always relative
to the threshold — never identically zero.

For one cross-section and a given alpha, this script:
  1. Loads the yearly quantile-normalized chunks.
  2. For every (year, month) and every characteristic permutation
     (3**tree_depth = 81 trees at depth 4), rebuilds the BFS of fuzzy
     weight vectors down to the leaves.
  3. Counts dead firms per leaf per month, averages across months.
  4. Aggregates across the 16 leaves per permutation and across the
     81 permutations.
  5. (Optional) reads `Selected_Ports_<K>.csv` from the matching
     fuzzy alpha grid dir to mark which leaves the K-portfolio SDF
     actually uses, so we can compare 'all leaves' vs 'used leaves'.

Output:
    plots/abalation/abalation1_<subdir>_dead_stocks_alpha<A>_K<K>.csv
        — one row per (perm, leaf): avg dead count, avg total firms,
          dead-pct, and a flag `selected_in_K` indicating membership
          in the K-portfolio EN selection.

Usage:
    PYTHONPATH=. python -u src/code/metrics/dead_stocks_diagnostic.py
    PYTHONPATH=. python -u src/code/metrics/dead_stocks_diagnostic.py --alpha 100 --K 22 --threshold 0.001
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from src.code import utils
from src.code.portfolio_creation.tree_portfolio_creation.fuzzy_tree_split import (
    fuzzy_split,
)
from src.code.portfolio_creation.tree_portfolio_creation.step2_generate_tree_portfolios_all_levels_char_minmax import (
    expand_grid,
)


def _depth4_leaf_weights(df_m, feat_list, alpha):
    """Return list of 16 weight vectors, one per depth-4 leaf, in BFS order
    (low-first). Mirrors `fuzzy_tree_month` but stops once the leaf level
    is reached and returns only that level."""
    n = len(df_m)
    level = [np.ones(n, dtype=np.float64)]
    for d in range(4):
        x = df_m[feat_list[d]].to_numpy(dtype=np.float64)
        nxt = []
        for w_parent in level:
            w_hi, w_lo, _ = fuzzy_split(x, w_parent, alpha=alpha)
            nxt.append(w_lo)
            nxt.append(w_hi)
        level = nxt
    return level


def _leaf_path_label(li):
    """Map BFS leaf index 0..15 -> '11111'..'12222' matching the cnames
    used by step2 / step3. BFS order with low-first appends gives
    binary-from-MSB, low=1, high=2, plus the leading '1' root marker."""
    bits = format(li, "04b")
    return "1" + "".join("2" if b == "1" else "1" for b in bits)


def _decode(perm_ids, leaf_path, feats):
    """'d=4 | LME=lo > OP=hi > LME=lo > Inv=lo' for human-readable output."""
    chars = [feats[i - 1] for i in perm_ids]
    dirs = ["lo" if c == "1" else "hi" for c in leaf_path[1:]]
    return "d=4 | " + " > ".join(f"{c}={d}" for c, d in zip(chars, dirs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat1", type=int, default=utils.FEAT1)
    ap.add_argument("--feat2", type=int, default=utils.FEAT2)
    ap.add_argument("--alpha", type=str, default="50",
                    help="Fuzzy alpha label matching the directory suffix.")
    ap.add_argument("--K", type=int, default=25,
                    help="K from the EN selection used to flag which leaves"
                         " the SDF actually uses (does not affect the count).")
    ap.add_argument("--threshold", type=float, default=1e-3,
                    help="Dead-stock cutoff on per-firm leaf weight (default 0.001 = 0.1%%).")
    ap.add_argument("--y-min", type=int, default=1964)
    ap.add_argument("--y-max", type=int, default=2016)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    feats_list = utils.FEATS_LIST
    feats = ["LME", feats_list[args.feat1 - 1], feats_list[args.feat2 - 1]]
    subdir = "_".join(feats)
    chunks_dir = os.path.join(utils.PY_DATA_CHUNK_DIR, subdir)
    n_perms = len(feats) ** 4
    feat_list_id_k = expand_grid(len(feats), 4)

    # Per-leaf running sums.
    dead_sum = np.zeros((n_perms, 16), dtype=np.int64)
    n_obs = 0  # total month-firm-pair observations across all months
    total_firms_running = 0  # for "avg firms per month" reporting
    months_seen = 0

    alpha_f = float(args.alpha)
    print(f"[dead-stocks] subdir={subdir} alpha={alpha_f} threshold={args.threshold}",
          flush=True)
    t0 = time.time()
    for y in range(args.y_min, args.y_max + 1):
        df_y = pd.read_csv(os.path.join(chunks_dir, f"y{y}.csv"))
        for m in range(1, 13):
            df_m = df_y.loc[df_y["mm"] == m, :]
            n = len(df_m)
            if n == 0:
                continue
            months_seen += 1
            total_firms_running += n
            for k in range(n_perms):
                perm_ids = feat_list_id_k[k]
                fl = [feats[i - 1] for i in perm_ids]
                leaves = _depth4_leaf_weights(df_m, fl, alpha_f)
                for li, w in enumerate(leaves):
                    dead_sum[k, li] += int((w < args.threshold).sum())
        if y % 5 == 0:
            print(f"  ...year {y} ({time.time() - t0:.1f}s elapsed)", flush=True)

    avg_dead = dead_sum / months_seen
    avg_firms_per_month = total_firms_running / months_seen
    avg_dead_pct = 100.0 * avg_dead / avg_firms_per_month

    # Membership in the EN-selected K portfolios.
    selected_ids = set()
    if args.K is not None:
        sel_path = os.path.join(
            utils.PY_FUZZY_TREE_GRID_DIR, f"{subdir}_a{args.alpha}",
            f"Selected_Ports_{args.K}.csv",
        )
        if os.path.isfile(sel_path):
            selected_ids = set(pd.read_csv(sel_path, nrows=0).columns)
            print(f"[dead-stocks] selected K={args.K} portfolio ids loaded from "
                  f"{sel_path} ({len(selected_ids)} ids)", flush=True)
        else:
            print(f"[dead-stocks] WARNING: {sel_path} not found — `selected_in_K` will be False everywhere",
                  flush=True)

    rows = []
    for k in range(n_perms):
        perm_ids = feat_list_id_k[k]
        perm_id = "".join(str(i) for i in perm_ids)
        for li in range(16):
            leaf_path = _leaf_path_label(li)
            pid = f"{perm_id}.{leaf_path}"
            rows.append({
                "portfolio_id": pid,
                "perm_id": perm_id,
                "leaf_path": leaf_path,
                "decoded": _decode(perm_ids, leaf_path, feats),
                "avg_dead_count": avg_dead[k, li],
                "avg_dead_pct": avg_dead_pct[k, li],
                "selected_in_K": pid in selected_ids,
            })
    df = pd.DataFrame(rows)

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    out_path = args.out or os.path.join(
        project_root, "plots", "abalation",
        f"abalation1_{subdir}_dead_stocks_alpha{args.alpha}_K{args.K}.csv",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    # Summary.
    print(f"\n[dead-stocks] elapsed: {time.time() - t0:.1f}s, months={months_seen}, "
          f"avg firms / month = {avg_firms_per_month:.1f}")
    print(f"[dead-stocks] threshold = {args.threshold} (firm w < threshold = 'dead')")
    print(f"\nALL DEPTH-4 LEAVES ({n_perms * 16} = {n_perms} perms x 16 leaves)")
    print(f"  avg dead count per leaf per month : {avg_dead.mean():.1f}")
    print(f"  avg dead %    per leaf per month : {avg_dead_pct.mean():.1f}%")
    print(f"  median        per leaf           : {np.median(avg_dead):.1f}")
    print(f"  10/90 pct     per leaf           : {np.percentile(avg_dead, 10):.1f} / {np.percentile(avg_dead, 90):.1f}")
    print(f"  min / max     per leaf           : {avg_dead.min():.1f} / {avg_dead.max():.1f}")

    if selected_ids:
        used = df[df["selected_in_K"]]
        print(f"\nLEAVES USED IN K={args.K} SDF (depth-4 only, others ignored): {len(used)}")
        if len(used) > 0:
            print(f"  avg dead count : {used['avg_dead_count'].mean():.1f}")
            print(f"  avg dead pct   : {used['avg_dead_pct'].mean():.1f}%")
            print(used[["portfolio_id", "decoded", "avg_dead_count",
                       "avg_dead_pct"]].to_string(index=False))

    print(f"\nwrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
