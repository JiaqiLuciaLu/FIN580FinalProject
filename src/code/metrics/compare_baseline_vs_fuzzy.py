"""Side-by-side comparison of hard-split vs fuzzy AP-Tree Sharpe ratios.

Reads `SR_N.csv` from the hard-split pipeline (`PY_TREE_GRID_DIR/<subdir>/`)
and from every fuzzy alpha directory it finds under `PY_FUZZY_TREE_GRID_DIR`
matching `<subdir>_a<alpha>/`, then writes a tidy CSV with one row per
(K, fold) and one column per pipeline (`baseline_SR`, `fuzzy_a5_SR`,
`fuzzy_a10_SR`, ...).

`SR_N.csv` layout (both pipelines):
    columns (after header)  : K - mink, where mink=5 by default
    row 1 of data           : train_SR per K
    row 2 of data           : valid_SR per K  (CV-fold averaged)
    row 3 of data           : test_SR  per K

Default output: `<project_root>/plots/abalation/
abalation1_<subdir>_baseline_vs_fuzzy_alpha_sweep.csv` — alongside the
companion K-selected-portfolios comparison.

Usage:
    PYTHONPATH=. python -u src/code/metrics/compare_baseline_vs_fuzzy.py
"""

import argparse
import os
import re

import pandas as pd

from src.code import utils


FOLD_NAMES = ["train", "valid", "test"]
# Match `<subdir>_a<label>` where label is any word-char/dot string —
# accommodates `a50`, `a50.5`, and `a50_dt0001` (manually pruned variant).
_ALPHA_DIR_RE = re.compile(r"^(?P<sub>.+)_a(?P<alpha>[\w.]+)$")


def _load_sr_n(path, mink=5):
    df = pd.read_csv(path)
    df.index = FOLD_NAMES[: len(df)]
    df.columns = [int(c) + mink for c in df.columns]
    return df


def _alpha_label(alpha_str):
    return f"a{alpha_str}"


def _discover_fuzzy_alpha_dirs(fuzzy_root, subdir):
    """Find every <subdir>_a<alpha>/ child of fuzzy_root that has SR_N.csv."""
    if not os.path.isdir(fuzzy_root):
        return {}
    out = {}
    for name in os.listdir(fuzzy_root):
        m = _ALPHA_DIR_RE.match(name)
        if not m or m.group("sub") != subdir:
            continue
        path = os.path.join(fuzzy_root, name, "SR_N.csv")
        if os.path.isfile(path):
            out[m.group("alpha")] = path
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat1", type=int, default=utils.FEAT1)
    ap.add_argument("--feat2", type=int, default=utils.FEAT2)
    ap.add_argument("--mink", type=int, default=5)
    ap.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: <project_root>/plots/abalation/"
        "abalation1_<subdir>_baseline_vs_fuzzy_alpha_sweep.csv)",
    )
    ap.add_argument("--skip-plot", action="store_true",
                    help="Skip the best-K summary PNG (still writes the CSVs).")
    args = ap.parse_args()

    subdir = utils.char_subdir(args.feat1, args.feat2)

    base = _load_sr_n(
        os.path.join(utils.PY_TREE_GRID_DIR, subdir, "SR_N.csv"), mink=args.mink
    )

    alpha_dirs = _discover_fuzzy_alpha_dirs(utils.PY_FUZZY_TREE_GRID_DIR, subdir)
    if not alpha_dirs:
        # Fallback: legacy unsuffixed dir for the case where only one alpha exists.
        legacy = os.path.join(utils.PY_FUZZY_TREE_GRID_DIR, subdir, "SR_N.csv")
        if os.path.isfile(legacy):
            alpha_dirs[""] = legacy
    fuzz_by_alpha = {
        a: _load_sr_n(p, mink=args.mink) for a, p in alpha_dirs.items()
    }
    def _alpha_sort_key(s):
        # Numeric labels first (sorted as floats), non-numeric (e.g. "50_dt0001")
        # last (sorted lexicographically). Keeps the natural α-sweep ordering
        # while still admitting variant labels.
        if s == "":
            return (0, 0.0)
        try:
            return (0, float(s))
        except ValueError:
            return (1, s)

    sorted_alphas = sorted(fuzz_by_alpha.keys(), key=_alpha_sort_key)

    rows = []
    for fold in FOLD_NAMES:
        if fold not in base.index:
            continue
        for K in base.columns:
            row = {"K": K, "fold": fold, "baseline_SR": base.at[fold, K]}
            for a in sorted_alphas:
                f = fuzz_by_alpha[a]
                col_name = f"fuzzy_{_alpha_label(a)}_SR" if a else "fuzzy_SR"
                if K in f.columns and fold in f.index:
                    row[col_name] = f.at[fold, K]
                    row[col_name.replace("_SR", "_delta")] = (
                        f.at[fold, K] - base.at[fold, K]
                    )
                else:
                    row[col_name] = None
                    row[col_name.replace("_SR", "_delta")] = None
            rows.append(row)
    out_df = pd.DataFrame(rows)

    # Default output: <project_root>/plots/abalation/. Compute project root
    # from this file's location: src/code/metrics/<this>.py -> 3 levels up.
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    out_path = args.out or os.path.join(
        project_root, "plots", "abalation",
        f"abalation1_{subdir}_baseline_vs_fuzzy_alpha_sweep.csv",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"wrote {out_path}  ({len(out_df)} rows, alphas={sorted_alphas})")

    # K snapshot.
    snapshot_K = [k for k in [10, 20, 40, 50] if k in base.columns]
    snap = out_df[out_df["K"].isin(snapshot_K)].copy()
    sr_cols = [c for c in snap.columns if c.endswith("_SR")]
    pivot = snap.pivot_table(index="K", columns="fold", values=sr_cols)
    print("\nSR snapshot (rounded):")
    print(pivot.round(4).to_string())

    # Best-K-via-validation summary: one row per pipeline (baseline + each
    # fuzzy alpha). For each pipeline we pick the K that maximises valid_SR
    # (the principled CV choice) and report the train/valid/test triple at
    # that K. Saved alongside the wide CSV.
    summary = _best_K_via_validation(base, fuzz_by_alpha, sorted_alphas)
    summary_path = os.path.join(
        os.path.dirname(out_path),
        f"abalation1_{subdir}_best_K_summary.csv",
    )
    summary.to_csv(summary_path, index=False)
    print(f"\nbest-K (via valid_SR) summary -> {summary_path}")
    print(summary.round(4).to_string(index=False))

    if not args.skip_plot:
        plot_path = summary_path.replace(".csv", ".png")
        _plot_best_K_summary(summary, plot_path, subdir)
        print(f"plot -> {plot_path}")


def _plot_best_K_summary(summary, plot_path, subdir):
    """Grouped bar chart of train / valid / test SR per pipeline at K*."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    folds = ["train_SR", "valid_SR", "test_SR"]
    fold_labels = ["Train", "Valid (CV)", "Test"]
    fold_colors = ["#9aa6b2", "#5a8fb8", "#1f4e79"]

    pipelines = summary["pipeline"].tolist()
    x = np.arange(len(pipelines))
    width = 0.27

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(pipelines)), 4.6))
    for i, (col, label, color) in enumerate(zip(folds, fold_labels, fold_colors)):
        vals = summary[col].astype(float).values
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    # Baseline test SR as a dashed reference line.
    if "baseline" in pipelines:
        base_test = float(summary.loc[summary["pipeline"] == "baseline", "test_SR"].iloc[0])
        ax.axhline(base_test, color=fold_colors[2], linestyle=":", linewidth=1, alpha=0.5)

    # K* annotated under each pipeline name.
    xtick_labels = [
        f"{p}\nK*={int(k)}" if k is not None else p
        for p, k in zip(pipelines, summary["best_K"].tolist())
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels)
    ax.set_ylabel("Sharpe ratio")
    ax.set_title(f"AP-Trees ablation 1 — best-K SR per pipeline ({subdir})")
    ax.set_ylim(0, max(0.05 + summary[folds].max().max(), 1.4))
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def _best_K_via_validation(base_df, fuzz_by_alpha, sorted_alphas):
    """Return one row per pipeline with K* chosen to maximise valid_SR
    and the resulting (train, valid, test) SR triple at K*.

    Columns: pipeline, best_K, train_SR, valid_SR, test_SR.
    """
    rows = []

    def _row(label, df):
        if "valid" not in df.index:
            return None
        valid = df.loc["valid"]
        if valid.dropna().empty:
            return {"pipeline": label, "best_K": None,
                    "train_SR": None, "valid_SR": None, "test_SR": None}
        best_K = int(valid.idxmax())
        return {
            "pipeline": label,
            "best_K": best_K,
            "train_SR": df.at["train", best_K] if "train" in df.index else None,
            "valid_SR": valid[best_K],
            "test_SR": df.at["test", best_K] if "test" in df.index else None,
        }

    rows.append(_row("baseline", base_df))
    for a in sorted_alphas:
        label = f"fuzzy_a{a}" if a else "fuzzy"
        rows.append(_row(label, fuzz_by_alpha[a]))
    return pd.DataFrame([r for r in rows if r is not None])


if __name__ == "__main__":
    main()
