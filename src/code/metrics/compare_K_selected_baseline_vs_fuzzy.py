"""Side-by-side comparison of K=N elastic-net selected portfolios — baseline
hard-split vs fuzzy α=A — for the LME_OP_Investment cross-section.

Reads `Selected_Ports_{K}.csv` and `Selected_Ports_Weights_{K}.csv` from:
    baseline:  PY_TREE_GRID_DIR / <subdir>/
    fuzzy:     PY_FUZZY_TREE_GRID_DIR / <subdir>_a<alpha>/

Decodes each `portfolio_id` (`<perm>.<path>`) into a human-readable split
chain (`LME=lo > OP=hi > ...`), pairs the two selections by id, and writes
a single merged CSV. Default output lands under `plots/abalation/` at
the project root.

Usage:
    PYTHONPATH=. python -u src/code/metrics/compare_K_selected_baseline_vs_fuzzy.py
    PYTHONPATH=. python -u src/code/metrics/compare_K_selected_baseline_vs_fuzzy.py --alpha 100 --K 20
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from src.code import utils


def _decode(col: str, feats: list[str]) -> str:
    """Decode `<perm>.<path>` into 'd=D | feat1=lo > feat2=hi > ...'.

    Path encoding (see `step2_generate_tree_portfolios_all_levels_char_minmax.py`'s
    `CNAMES_*`): the first char is a constant root marker ('1'), the
    remaining chars are 1=low / 2=high choices at each split. So a length-
    (D+1) path string corresponds to a depth-D node.
    """
    perm, path = col.split(".")
    depth = len(path) - 1
    if depth == 0:
        return "d=0 | root"
    perm_chars = [feats[int(c) - 1] for c in perm[:depth]]
    dirs = ["lo" if c == "1" else "hi" for c in path[1:]]
    return f"d={depth} | " + " > ".join(f"{f}={d}" for f, d in zip(perm_chars, dirs))


def _load_selection(grid_dir: str, K: int, feats: list[str]) -> pd.DataFrame:
    ports = pd.read_csv(os.path.join(grid_dir, f"Selected_Ports_{K}.csv"))
    w = pd.read_csv(os.path.join(grid_dir, f"Selected_Ports_Weights_{K}.csv"))["x"].values
    mu = ports.mean() * 12
    sd = ports.std() * np.sqrt(12)
    df = pd.DataFrame(
        {
            "portfolio_id": ports.columns,
            "EN_weight": w,
            "ann_mean(%)": (mu * 100).round(2).values,
            "ann_std(%)": (sd * 100).round(2).values,
            "ann_SR": (mu / sd).round(3).values,
            "decoded": [_decode(c, feats) for c in ports.columns],
        }
    )
    return df.reindex(np.argsort(-np.abs(w))).reset_index(drop=True)


def _merge_side_by_side(base: pd.DataFrame, fuzz: pd.DataFrame, alpha_label: str) -> pd.DataFrame:
    rename = lambda df, p: df.rename(  # noqa: E731
        columns={c: f"{p}_{c}" if c != "portfolio_id" else c for c in df.columns}
    )
    b = rename(base, "baseline")
    f = rename(fuzz, alpha_label)
    m = pd.merge(b, f, on="portfolio_id", how="outer")
    # Either side has decoded; collapse into one column.
    decoded_cols = [c for c in m.columns if c.endswith("_decoded")]
    m["decoded"] = m[decoded_cols].bfill(axis=1).iloc[:, 0]
    m = m.drop(columns=decoded_cols)
    # Sort by max absolute weight across the two models.
    bw = m["baseline_EN_weight"].abs().fillna(0)
    fw = m[f"{alpha_label}_EN_weight"].abs().fillna(0)
    m = (
        m.assign(_max_w=np.maximum(bw, fw))
        .sort_values("_max_w", ascending=False)
        .drop(columns=["_max_w"])
        .reset_index(drop=True)
    )
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat1", type=int, default=utils.FEAT1)
    ap.add_argument("--feat2", type=int, default=utils.FEAT2)
    ap.add_argument("--alpha", type=str, default="50",
                    help="Fuzzy alpha label matching the directory suffix (e.g. '50' or '10').")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--out", default=None,
                    help="Output CSV path (default: plots/abalation/"
                         "abalation1_K<K>_baseline_vs_alpha<alpha>_selected_portfolios.csv)")
    args = ap.parse_args()

    feats_list = utils.FEATS_LIST
    feats = ["LME", feats_list[args.feat1 - 1], feats_list[args.feat2 - 1]]
    subdir = utils.char_subdir(args.feat1, args.feat2, feats_list)

    base_dir = os.path.join(utils.PY_TREE_GRID_DIR, subdir)
    fuzz_dir = os.path.join(utils.PY_FUZZY_TREE_GRID_DIR, f"{subdir}_a{args.alpha}")
    for d, label in [(base_dir, "baseline"), (fuzz_dir, f"fuzzy α={args.alpha}")]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"{label} grid dir missing: {d}")

    base = _load_selection(base_dir, args.K, feats)
    fuzz = _load_selection(fuzz_dir, args.K, feats)

    print("=" * 100)
    print(f"BASELINE (hard-split) — K={args.K} selected portfolios")
    print("=" * 100)
    print(base.to_string(index=False))
    print()
    print("=" * 100)
    print(f"FUZZY α={args.alpha} — K={args.K} selected portfolios")
    print("=" * 100)
    print(fuzz.to_string(index=False))

    b_ids, f_ids = set(base["portfolio_id"]), set(fuzz["portfolio_id"])
    common = sorted(b_ids & f_ids)
    only_b = sorted(b_ids - f_ids)
    only_f = sorted(f_ids - b_ids)
    print()
    print("=" * 100)
    print("OVERLAP / DIFFERENCE")
    print("=" * 100)
    print(f"common picks      ({len(common)}): {common}")
    print(f"baseline only     ({len(only_b)}): {only_b}")
    print(f"fuzzy α={args.alpha} only      ({len(only_f)}): {only_f}")
    print(f"sum |w| baseline  : {base['EN_weight'].abs().sum():.2f}")
    print(f"sum |w| fuzzy α={args.alpha}: {fuzz['EN_weight'].abs().sum():.2f}")

    merged = _merge_side_by_side(base, fuzz, alpha_label=f"fuzzy_a{args.alpha}")

    # Default output: <project_root>/plots/abalation/. Compute project root
    # from this file's location: src/code/metrics/<this>.py -> 3 levels up.
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    out_path = args.out or os.path.join(
        project_root, "plots", "abalation",
        f"abalation1_K{args.K}_baseline_vs_alpha{args.alpha}_selected_portfolios.csv",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}  ({len(merged)} unique portfolios across both)")


if __name__ == "__main__":
    main()
