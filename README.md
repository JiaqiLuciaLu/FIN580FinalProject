# FIN580 Final Project — AP-Trees

Replication and extension of Bryzgalova, Pelger & Zhu (2020), *"Forest Through the
Trees: Building Cross-Sections of Stock Returns."* The paper builds **Asset
Pricing Trees** — conditional median sorts of firms into tree-leaf portfolios,
pruned by an elastic-net SDF. We port the authors' R pipeline to Python,
replicate the headline tables/figures, and run an ablation that swaps the hard
median split for a soft (sigmoid) split.

**Disclaimer:** We used Claude to generate/implement the project code base.

## Code structure

```
src/
  code/
    utils.py                       # paths, constants, FEATS_LIST
    portfolio_creation/            # Steps 1–4: panels → trees → filtered leaves
      data_prep.py, characteristics.py, splits.py, tree_construction.py
      tree_portfolio_creation/, double_sort_*, triple_sort_*, longshort_decile_*
    ap_pruning/                    # LARS-EN + (λ₀, λ₂) grid search
      lasso.py, ap_pruning.py, lasso_valid_par_full.py
    metrics/                       # SR, SDF/FF regressions, XS-R², diagnostics
      pick_best_lambda.py, sr_n.py, sdf_timeseries_regressions.py
      compare_baseline_vs_fuzzy.py, dead_stocks_diagnostic.py, ...
    ablation/                      # long-only SDF variant (LARS-positive)
    plots.py                       # Figure 6 / 10 / 11 / 12 generators
  notebooks/scratch/               # end-to-end runners (compute_table1.py, run_sigmoid_*.py, ...)
  reference_code/                  # original R implementation, kept for diffing

slurm/                             # sbatch wrappers for compute-heavy phases
docs/   status.md, abalation_design.md, exploration.md
plots/  Figure6/, Figure10/, table1/, abalation/   # generated outputs
```

Raw CRSP + author CSVs live at `/scratch/network/jl6134/COLLAB/FIN580/data/raw/`;
intermediate and final outputs land in `…/data/processed/` (both outside git).

## Ablation (one-liner)

We replace the hard median split `ntile(x, 2)` with a per-firm sigmoid
`σ(α(x − m))` so every firm carries fractional membership in every leaf, then
sweep the steepness α; α≈50 with a 10–20 % dead-weight prune gives the best
test Sharpe, slightly beating the hard-split baseline by reducing EN overfit on
the soft tail. Design lives in `docs/abalation_design.md`; numerical write-up
in `plots/abalation/observations.md`.

## Where to look

| Output | Path |
|---|---|
| Replication tables (Table 1) | `plots/table1/table1_comparison.csv` |
| Figure 6 (SR vs K) | `plots/Figure6/SRwithXSF_{10,40}.png` |
| Figure 10 (heatmaps) | `plots/Figure10/LME_OP_Investment_*.png` |
| Ablation curves & summaries | `plots/abalation/abalation1_*.{csv,png}` |
| Ablation observations | `plots/abalation/observations.md` |
| End-to-end runners | `src/notebooks/scratch/` (`compute_table1*.py`, `run_sigmoid_*.py`, `run_table1_baseline.py`) |
| Phase history & validation log | `docs/status.md` |

## Environment

```bash
module load anaconda3/2025.6
conda activate FIN580   # Python 3.11; numpy/pandas/scipy/sklearn/statsmodels
```

Long jobs go through `slurm/*.sbatch`, not the login node.
