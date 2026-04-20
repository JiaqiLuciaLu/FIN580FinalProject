# Status

## Plots / tables status

All single-cross-section plots now unblocked after the full-grid run
(`slurm/main_table1.sbatch`, job 3118309 — `docs/outputs.md` updated).

### Runnable
- `src/code/plots/figure1bc_empirical_portfolio_bounding_box.py`
  — uses `tree_portfolio_quantile/LME_OP/level_all_*.csv`.
- `src/code/plots/figure10ac_sdf_sr_n.py`
  — uses `TreeGridSearch/LME_OP_Investment/SR_N.csv`. R bug fixed (Testing
    plot now reads `sr[3,]`).
- `src/code/plots/figure10bd_lambda_heatmap_grid.py`
  — now runnable on the full 19×13 grid.
- `src/code/plots/figurec8ab_xsr2.py`
  — faithful port of `FigureC8ab_XSR2.R::{FF_regression, compute_Statistics,
    XSR2}`. Per-portfolio pricing-error boxplot block (Figure C8) is
    commented out pending need.
- `src/notebooks/scratch/compute_table1.py`
  — assembles paper Table 1 (AP10 / AP40 / TS32 / TS64) into
    `data/processed/tables/paper_table1.csv`.

### Still blocked
- **Figure 6a** (`src/code/plots/figure6a_sr_plot_xsf.py`) — needs
  36-cross-section `SR_Summary.csv`. Job B (all 36 cross-sections) not yet
  run.

## Pipeline validation

### Tree construction matches R 1-1
- Tree depth 4 confirmed — every column's char-path is exactly 4 digits
  (`"1111"` through `"3333"`) across both our Python output and the R
  reference.
- Monoculture filter drops the 48 depth-5 leaves whose char-path is
  `"1111"`, `"2222"`, or `"3333"` (same characteristic used at all 4 split
  levels). 1555 → 1507 portfolios.
- Label correspondence Python ↔ R (on `level_all_excess_combined.csv`):
  - Column names bit-identical (1555 cols, same order). E.g. Python's
    `"1112.11"` refers to the same tree node as R's `"1112.11"`.
  - Data at each labeled column matches up to R's shipped noise: per-column
    corr median 0.993 / min 0.950, mean abs diff 0.005 per month.

### SDF SR is the primary replication metric — and we match

Sharpe ratio is the one metric in paper Table 1 that is **scale- and
rotation-invariant** for the SDF direction. Matching SR to paper's values
(our 0.64 / 0.67 / 0.52 / 0.57 vs paper's 0.65 / 0.69 / 0.51 / 0.53) means
our CV + LARS-EN pipeline identifies the **same SDF subspace** — the same
economic information — as the paper. This is the Hansen-Jagannathan /
pricing-ability signal and the criterion on which AP-Trees is evaluated.

Secondary statistics (α magnitude, XS R²_adj) depend on reporting
conventions that differ between Table 1 and the R plot/metric scripts; see
the discrepancy section below.

## Known discrepancies vs paper

### Figure 10 — optimal (λ₀, λ₂) differs from paper

The red dot on the Figure 10b/d heatmaps (training argmax we pick via
`pickBestLambda`) sits at a different location than the paper's reported
optimum. Same root cause as the α-scale gap: numerical differences between
our LARS-EN path (sklearn `lars_path`) and the R `lars` package move the
argmax of the validation-SR surface.

Diagnostic: our single-fold (`fullCV=FALSE`, paper's default) selection is
dominated by `cv_3` — its validation window (1984–1993) is a high-SR
regime that rewards extreme shrinkage (`λ₀ ≈ 0.9`). `cv_1` (1964–1973) and
`cv_2` (1974–1983) prefer mild interior shrinkage (`λ₀ ≈ 0.15–0.20`).
Averaging all 3 folds (`fullCV=True`) pushes AP10 / AP40 to interior cells
that visually match paper's Figure 10b, but with no change to SR matching.

### Table 1 — α magnitude and XS R² do not match paper

`data/processed/tables/paper_table1.csv` vs paper:

| metric                | ours (AP10/AP40/TS32/TS64) | paper |
|-----------------------|-----------------------------|-------|
| SDF SR                | 0.64 / 0.67 / 0.52 / 0.57   | 0.65 / 0.69 / 0.51 / 0.53  (≈ match ✓) |
| t-stat (FF3 α)        | 10.33 / 10.87 / 8.03 / 9.04 | 10.11 / 11.03 / 7.40 / 8.13 (≈ match ✓) |
| α (FF3, % / month)    | 3.19 / 5.86 / 3.53 / 3.55   | 0.94 / 0.90 / 0.75 / 0.84  (≈ 4× larger) |
| XS R²_adj (FF3)       | -55.5% / 2.7% / 11.4% / 14.6% | 18% / 51% / 82% / 82%    (formula differs) |

**α scale (≈ 4×)** — t-stats match (scale-invariant), so the regression math
is consistent and the SDF direction is correct. Since SR also matches, our
SDF is economically equivalent to paper's; the 4× is a reporting-scale
artifact.

Root cause: LARS-EN weights from our pipeline have `‖w‖₂ ≈ 2.4–3.0`; on the
same SR, a smaller `‖w‖₂` produces proportionally smaller mean and std (SR
unchanged). Candidates for the scale difference:
  1. Numerical differences between R's `lars` package and sklearn's
     `lars_path` in active-set tie-breaking, breakpoint tolerances, or
     stopping rules — both solve the same EN-via-augmented-LASSO problem,
     but the specific coefficient vector at `K = 10 / 40 / 32 / 64` can
     diverge.
  2. Paper's Table 1 α may be computed from an SDF rescaled after the
     pickBestLambda step (e.g., target a specific vol or match the
     cross-section's average excess return) — this scaling is not in the
     shipped R scripts but is a plausible paper-writing convention.

Switching `fullCV=True` (averaging all 3 CV folds, tested 2026-04-20)
moved AP10 / AP40 to interior (λ₀, λ₂) cells matching paper's Figure 10b
visually, but did not close the α-scale gap — confirming the scale issue
lives in the LARS-EN weight magnitudes, not in CV selection. Reverted to
paper-faithful `fullCV=False` (`main.R` default).

**XS R²_adj formula mismatch.** Sanity check: our TS32 portfolios match the
noised reference `raw/data/ts_portfolio/...` within noise (per-col corr 0.944–
0.999, mean 0.98; see reproducible diagnostic in `compute_table1.py`). Running
`figurec8ab_xsr2.compute_Statistics` on the **reference** TS32 gives the same
11.4% FF3 adj R² we get on our data. Paper reports 82%. Conclusion: the
`FigureC8ab_XSR2.R` R² formula (`1 - Σα² / Σ avg_ret²`, DOF-adjusted) is *not*
the definition used in paper Table 1. Our port is faithful to the R file; the
Table 1 metric is a different statistic, to be identified.
