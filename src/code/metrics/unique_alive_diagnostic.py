"""Unique-alive-firm diagnostic for the K-portfolio fuzzy SDF.

For each fuzzy variant under `PY_FUZZY_TREE_GRID_DIR` matching
`<subdir>_a<label>/` that has a `Selected_Ports_<K>.csv`, this script:

  1. Reads the K selected portfolio ids (`<perm>.<path>` codes).
  2. Replays the fuzzy_split BFS for each unique permutation, per
     (year, month), to recover the per-firm weight vector at every
     selected node (any depth, not just leaves).
  3. For each portfolio, marks firms with w >= threshold as 'alive'.
  4. Per month, takes the union across all K portfolios -> unique
     alive firms used by the SDF that month.
  5. Averages across months.

Output CSV (one row per variant): avg unique alive per month, avg
alive-per-portfolio, depth mix of the K selection, total months seen.

Usage:
    PYTHONPATH=. python -u src/code/metrics/unique_alive_diagnostic.py
    PYTHONPATH=. python -u src/code/metrics/unique_alive_diagnostic.py --K 25 --threshold 0.001
"""

from __future__ import annotations

import argparse
import os
import re
import time

import numpy as np
import pandas as pd

from src.code import utils
from src.code.portfolio_creation.tree_portfolio_creation.fuzzy_tree_split import (
    fuzzy_split,
)


_ALPHA_RE = re.compile(r"^(?P<sub>.+)_a(?P<label>[\w.]+)$")


def _parse_pid(pid):
    perm, path = pid.split(".")
    depth = len(path) - 1                         # path[0] is the root marker
    return perm, depth, path


def _bfs_index(path):
    """BFS index of node `path` within its depth level.

    Drop the leading root marker; treat 1->0 (low) and 2->1 (high) MSB-first.
    Matches the order in which `fuzzy_tree_month` appends (low, high) per parent.
    """
    if len(path) == 1:
        return 0
    bits = "".join("0" if c == "1" else "1" for c in path[1:])
    return int(bits, 2)


def _all_node_weights(df_m, feat_list, alpha, max_depth):
    """levels[d] = list of 2**d weight vectors at depth d (BFS, low-first)."""
    n = len(df_m)
    levels = [[np.ones(n, dtype=np.float64)]]
    for d in range(max_depth):
        x = df_m[feat_list[d]].to_numpy(dtype=np.float64)
        nxt = []
        for w_parent in levels[d]:
            w_hi, w_lo, _ = fuzzy_split(x, w_parent, alpha=alpha)
            nxt.append(w_lo)
            nxt.append(w_hi)
        levels.append(nxt)
    return levels


def _discover_variants(fuzzy_root, subdir, K):
    out = {}
    if not os.path.isdir(fuzzy_root):
        return out
    for name in os.listdir(fuzzy_root):
        m = _ALPHA_RE.match(name)
        if not m or m.group("sub") != subdir:
            continue
        sel = os.path.join(fuzzy_root, name, f"Selected_Ports_{K}.csv")
        if os.path.isfile(sel):
            out[m.group("label")] = name
    return out


def _alpha_value(label):
    """For 'a50' -> 50.0; for 'a50_dt0001' -> 50.0 (the digits before any '_')."""
    head = label.split("_", 1)[0]
    return float(head)


def _stats_one_variant(grid_dir, alpha_value, K, threshold, chunks_dir,
                      feats, y_min, y_max):
    sel_df = pd.read_csv(os.path.join(grid_dir, f"Selected_Ports_{K}.csv"), nrows=0)
    pids = list(sel_df.columns)

    # Group selected portfolios by permutation (so we replay each perm once).
    by_perm = {}
    depth_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for pid in pids:
        perm, depth, path = _parse_pid(pid)
        by_perm.setdefault(perm, []).append((depth, path, pid))
        depth_counts[depth] += 1

    n_unique_alive_per_month = []
    alive_per_port_running = {pid: 0 for pid in pids}
    months = 0
    total_firms = 0

    for y in range(y_min, y_max + 1):
        df_y = pd.read_csv(os.path.join(chunks_dir, f"y{y}.csv"))
        for m in range(1, 13):
            df_m = df_y.loc[df_y["mm"] == m, :]
            n = len(df_m)
            if n == 0:
                continue
            months += 1
            total_firms += n

            union_mask = np.zeros(n, dtype=bool)
            for perm, items in by_perm.items():
                feat_list = [feats[int(c) - 1] for c in perm]
                levels = _all_node_weights(df_m, feat_list, alpha_value, max_depth=4)
                for depth, path, pid in items:
                    idx = _bfs_index(path)
                    w = levels[depth][idx]
                    is_alive = w >= threshold
                    alive_per_port_running[pid] += int(is_alive.sum())
                    union_mask |= is_alive
            n_unique_alive_per_month.append(int(union_mask.sum()))

    avg_per_port = {p: alive_per_port_running[p] / max(months, 1) for p in pids}
    return {
        "label": None,                           # filled in by caller
        "alpha_value": alpha_value,
        "K": K,
        "threshold": threshold,
        "months": months,
        "avg_firms_per_month": total_firms / max(months, 1),
        "avg_unique_alive_per_month": float(np.mean(n_unique_alive_per_month)),
        "median_unique_alive_per_month": float(np.median(n_unique_alive_per_month)),
        "min_unique_alive_per_month": int(np.min(n_unique_alive_per_month)),
        "max_unique_alive_per_month": int(np.max(n_unique_alive_per_month)),
        "avg_alive_per_portfolio": float(np.mean(list(avg_per_port.values()))),
        "depth_count_d0": depth_counts[0],
        "depth_count_d1": depth_counts[1],
        "depth_count_d2": depth_counts[2],
        "depth_count_d3": depth_counts[3],
        "depth_count_d4": depth_counts[4],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat1", type=int, default=utils.FEAT1)
    ap.add_argument("--feat2", type=int, default=utils.FEAT2)
    ap.add_argument("--K", type=int, default=25)
    ap.add_argument("--threshold", type=float, default=1e-3)
    ap.add_argument("--y-min", type=int, default=1964)
    ap.add_argument("--y-max", type=int, default=2016)
    ap.add_argument("--variants", nargs="*", default=None,
                    help="Subset of fuzzy alpha labels to run (default: all discovered).")
    args = ap.parse_args()

    feats_list = utils.FEATS_LIST
    feats = ["LME", feats_list[args.feat1 - 1], feats_list[args.feat2 - 1]]
    subdir = "_".join(feats)
    chunks_dir = os.path.join(utils.PY_DATA_CHUNK_DIR, subdir)

    variants = _discover_variants(utils.PY_FUZZY_TREE_GRID_DIR, subdir, args.K)
    if args.variants:
        variants = {k: v for k, v in variants.items() if k in args.variants}
    if not variants:
        raise SystemExit(f"No variants found under {utils.PY_FUZZY_TREE_GRID_DIR} "
                         f"for subdir={subdir} with Selected_Ports_{args.K}.csv")

    rows = []
    for label, dir_name in sorted(variants.items()):
        alpha_val = _alpha_value(label)
        grid_dir = os.path.join(utils.PY_FUZZY_TREE_GRID_DIR, dir_name)
        print(f"\n[unique-alive] variant={label} alpha_value={alpha_val} dir={dir_name}",
              flush=True)
        t0 = time.time()
        s = _stats_one_variant(
            grid_dir=grid_dir, alpha_value=alpha_val,
            K=args.K, threshold=args.threshold,
            chunks_dir=chunks_dir, feats=feats,
            y_min=args.y_min, y_max=args.y_max,
        )
        s["label"] = label
        rows.append(s)
        print(f"  done in {time.time() - t0:.1f}s — "
              f"avg unique alive / month: {s['avg_unique_alive_per_month']:.1f}",
              flush=True)

    df = pd.DataFrame(rows)
    cols = ["label", "alpha_value", "K", "threshold", "months",
            "avg_firms_per_month", "avg_unique_alive_per_month",
            "median_unique_alive_per_month", "min_unique_alive_per_month",
            "max_unique_alive_per_month", "avg_alive_per_portfolio",
            "depth_count_d0", "depth_count_d1", "depth_count_d2",
            "depth_count_d3", "depth_count_d4"]
    df = df[cols]
    print("\n=== summary ===")
    print(df.to_string(index=False))

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    out = os.path.join(
        project_root, "plots", "abalation",
        f"abalation1_{subdir}_unique_alive_K{args.K}.csv",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
