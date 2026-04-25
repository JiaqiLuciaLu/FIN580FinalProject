"""Visualize and quantify how steep the fuzzy split is at different α.

Three artifacts saved under plots/abalation/:
  1. abalation1_alpha_sigmoid_curves.png  — σ(α(x − 0.5)) for x ∈ [0, 1] at
     each α (one line per value). Shows the per-split shape directly.
  2. abalation1_alpha_depth4_cumulative.png — distribution of leaf weights
     w_leaf = ∏σ(α·(x_i − m_i)) across 4 splits, assuming the firm sits at
     symmetric quantile distance δ from each parent's median. Shows how
     quickly cumulative weight saturates after 4 splits.
  3. abalation1_alpha_steepness_summary.csv — numerical stats per α:
     transition-zone width (where σ ∈ [0.01, 0.99]), fraction of uniform-
     distributed firms in the transition zone, expected alive fraction
     after depth-4 splits at thresholds 0.001 / 0.01 / 0.1.

Pure-Python; no SLURM, no dependence on running pipelines. Just produces
intuition for the α knob.

Usage:
    PYTHONPATH=. python -u src/code/metrics/alpha_steepness_diagnostic.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[10.0, 50.0, 100.0, 200.0])
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    out_dir = args.out_dir or os.path.join(project_root, "plots", "abalation")
    os.makedirs(out_dir, exist_ok=True)

    # ---------- numerical summary ----------
    rows = []
    for alpha in args.alphas:
        # Transition zone half-width on the (x - 0.5) axis: σ(α·δ) = 0.99
        # ⇒ α·δ = log(99) ≈ 4.595. So δ_99 = 4.595 / α. Symmetric for 0.01.
        delta_99 = float(np.log(99.0) / alpha)
        # In quantile space, x ∈ [0, 1], so the fraction of firms inside the
        # transition zone (σ ∈ [0.01, 0.99] at one split) is min(1, 2·δ_99)
        # since x − 0.5 ranges over [−0.5, 0.5].
        trans_frac_one_split = min(1.0, 2 * delta_99)

        # For depth-d cumulative weight, treat each split as the firm at
        # symmetric distance δ from the parent median, going to the high
        # side every time. Then w_leaf = σ(α·δ)^d. Compute the δ at which
        # w_leaf crosses {0.001, 0.01, 0.1, 0.5}.
        def delta_for_w(w_target, d=args.depth):
            # σ(αδ)^d = w  ⇒  σ(αδ) = w^(1/d)  ⇒  αδ = logit(w^(1/d))
            s = w_target ** (1.0 / d)
            if s <= 0 or s >= 1:
                return float("nan")
            return float(np.log(s / (1 - s)) / alpha)

        # Fraction of uniform-distributed firms (x ∈ [0, 1]) whose
        # one-side weight at a single split is >= threshold. With
        # σ(α·(x − 0.5)) >= τ ⇔ x − 0.5 >= logit(τ)/α (where logit(0.5)=0,
        # so for τ < 0.5 this gives a region on the high side).
        def frac_above(tau):
            if tau <= 0:
                return 1.0
            if tau >= 1:
                return 0.0
            cutoff = 0.5 + np.log(tau / (1 - tau)) / alpha
            return float(max(0.0, min(1.0, 1 - cutoff)))

        rows.append({
            "alpha": alpha,
            "transition_half_width_(σ_0.01→0.99)": round(delta_99, 5),
            "transition_zone_frac_one_split": round(trans_frac_one_split, 4),
            f"depth{args.depth}_w_leaf_at_δ=0.05": round(sigmoid(alpha * 0.05) ** args.depth, 4),
            f"depth{args.depth}_w_leaf_at_δ=0.10": round(sigmoid(alpha * 0.10) ** args.depth, 4),
            f"depth{args.depth}_w_leaf_at_δ=0.20": round(sigmoid(alpha * 0.20) ** args.depth, 4),
            "frac_alive_>=0.001_one_split": round(frac_above(0.001), 4),
            "frac_alive_>=0.01_one_split":  round(frac_above(0.01), 4),
            "frac_alive_>=0.10_one_split":  round(frac_above(0.10), 4),
            "frac_alive_>=0.50_one_split":  round(frac_above(0.50), 4),
        })
    summary = pd.DataFrame(rows)
    summary_path = os.path.join(out_dir, "abalation1_alpha_steepness_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSteepness summary table:")
    print(summary.to_string(index=False))
    print(f"\nwrote {summary_path}")

    # ---------- plot ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel 1: per-split sigmoid curves
    x = np.linspace(0.0, 1.0, 1001)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(args.alphas)))
    for alpha, c in zip(args.alphas, colors):
        s = sigmoid(alpha * (x - 0.5))
        ax1.plot(x, s, label=f"α={alpha:g}", color=c, linewidth=2)
    ax1.axvline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax1.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax1.set_xlabel("quantile-normalized characteristic x")
    ax1.set_ylabel("σ(α · (x − 0.5))  [high-side weight, one split]")
    ax1.set_title("Per-split sigmoid")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    # Panel 2: cumulative weight at depth d as function of distance-from-median δ
    delta = np.linspace(-0.5, 0.5, 1001)
    for alpha, c in zip(args.alphas, colors):
        # symmetric path: firm at distance δ from median at every split,
        # always taking the high-side branch -> cumulative weight σ(αδ)^d
        s = sigmoid(alpha * delta) ** args.depth
        ax2.plot(delta, s, label=f"α={alpha:g}", color=c, linewidth=2)
    ax2.axhline(0.001, color="red", linestyle="--", linewidth=1, alpha=0.5,
                label="dt=0.1%")
    ax2.axhline(0.01, color="orange", linestyle="--", linewidth=1, alpha=0.5,
                label="dt=1%")
    ax2.axhline(0.1, color="green", linestyle="--", linewidth=1, alpha=0.5,
                label="dt=10%")
    ax2.set_xlabel("symmetric distance from each parent median (δ)")
    ax2.set_ylabel(f"w_leaf = σ(α·δ)^{args.depth}")
    ax2.set_title(f"Cumulative leaf weight after depth={args.depth}")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-6, 1.1)
    ax2.legend(loc="lower right", ncol=2)
    ax2.grid(alpha=0.3, which="both")

    fig.suptitle("Fuzzy AP-Tree split steepness comparison")
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "abalation1_alpha_steepness_curves.png")
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
