# CLAUDE.md

Guidance for Claude Code working in this repo. Kept deliberately short; detailed phase history and validation results live in `docs/status.md`.

## Project

Replication and extension of **"Forest Through the Trees: Building Cross-Sections of Stock Returns"** (Bryzgalova, Pelger, Zhu, 2020). The paper introduces **Asset Pricing Trees (AP-Trees)** — conditional sorting via decision trees combined with elastic net pruning.

Goals: translate the authors' R implementation into Python, replicate core results (Sharpe ratios, pricing errors, cross-sectional R²), extend via a transformer encoder for top-k portfolio selection (replacing LASSO), and produce a written report.

## Environment

Conda env `FIN580` (Python 3.11) at `~/.conda/envs/FIN580`.

```bash
module load anaconda3/2025.6
conda activate FIN580
```

Installed: numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib, seaborn, jupyter. Add new packages with `conda install -n FIN580 <pkg>` or `pip install <pkg>`.

## Architecture

Everything lives under `src/`, mirroring the R reference pipeline. Shared data lives at `/scratch/network/jl6134/COLLAB/FIN580/data/raw/` (not in git).

```
src/
  code/
    utils.py                       # shared paths, constants, helpers
    portfolio_creation/
      data_prep.py                 # load_factors(), load_rf(), load_filtered_tree_portfolios()
    ap_pruning/
      lasso.py                     # lasso_en() — LARS-EN via augmented-matrix trick
      pruning.py                   # ap_pruning() — grid search over (λ₀, λ₂)
    metrics/
      metrics.py                   # pick_best_lambda(), pick_sr_n()
      regressions.py               # sdf_regression(), ff_regression() — FF3/FF5/XSF/FF11
    ablation/                      # (future: transformer encoder, alternative pruning)
  notebooks/                       # end-to-end pipeline runners
  reference_code/                  # original R implementation
plots/                             # figure outputs
output/                            # Python pipeline outputs (separate from R ground-truth)
```

- **`src/code/utils.py`** — paths (`DATA_RAW`, `TREE_GRID_DIR`, `PY_TREE_GRID_DIR`), constants (`FEATS_LIST`, `FEAT1=4`, `FEAT2=5`, `N_TRAIN_VALID=360`, `CV_N=3`, `KMIN=5`, `KMAX=50`), subdirectory helpers.
- **`src/code/portfolio_creation/data_prep.py`** — `load_factors()`, `load_rf()`, `load_filtered_tree_portfolios()`, `extract_depths()`. Mirrors R's `1_Portfolio_Creation/`.
- **`src/code/ap_pruning/lasso.py`** — `lasso_en(X, y, λ₂, kmin, kmax)`: LARS-EN path via augmented-matrix trick on `sklearn.linear_model.lars_path`. Mirrors R's `lasso.R`.
- **`src/code/ap_pruning/pruning.py`** — `ap_pruning(...)`: eigendecomp of Σ, σ̃/μ̃, LARS-EN per (λ₀, λ₂), writes `results_{cv_k,full}_l0_i_l2_j.csv`. Mirrors `AP_Pruning.R` + `lasso_valid_par_full.R`. Supports `n_workers` via `multiprocessing.Pool`.
- **`src/code/metrics/metrics.py`** — `pick_best_lambda(...)`, `pick_sr_n(...)`: reads grid CSVs, builds SR heatmaps, writes `Selected_Ports_K.csv`, `Selected_Ports_Weights_K.csv`, `SR_N.csv`. Mirrors `Pick_Best_Lambda.R` + `SR_N.R`.
- **`src/code/metrics/regressions.py`** — `sdf_regression(...)`, `ff_regression(...)`: FF3/FF5/XSF/FF11, writes `SDFTests/<subdir>/TimeSeriesAlpha.csv`. Mirrors `SDF_TimeSeries_Regressions.R`.
- **`src/code/ablation/`** — placeholder for future extensions (transformer encoder for top-k selection, alternative pruning strategies).

### Running the pipelines

```bash
PYTHONPATH=. python src/notebooks/run_phase_bc.py           # single-config (main_simplified.R)
PYTHONPATH=. python src/notebooks/run_phase_d.py            # full grid (main.R)
PYTHONPATH=. python src/notebooks/run_phase_d.py --skip-grid  # reuse 988 CSVs; rerun downstream only
```

## Current Status

Phases A–D complete (Python replication matches R end-to-end). **Next: Phase E** (figures 10a/c, 11, 12 + Paper Table 1 row).

See **`docs/status.md`** for phase-by-phase details, validation results, caveats, and the Phase E plan. Validation artifacts: `output/replication_results/*/run_manifest.json`.

## Data

Source: authors' CSVs at `/scratch/network/jl6134/COLLAB/FIN580/data/raw/`. Monthly equity returns, Jan 1964 – Dec 2016.

**10 firm characteristics** (FEATS_LIST): LME, BEME, r12_2, OP, Investment, ST_Rev, LT_Rev, AC, IdioVol, LTurnover.

**Sample split**: Training 1964–1983 (240mo), Validation 1984–1993 (120mo), Testing 1994–2016 (276mo).

**Current scope**: 3-char case LME_OP_Investment (FEAT1=4, FEAT2=5). Data for other 35 cross-sections is not available.

## Key methodology

- **AP-Tree splits**: binary at characteristic median. Depth-d tree → 2^d leaf portfolios.
- **Elastic net hyperparameters**: λ₀ (mean shrinkage), λ₂ (variance shrinkage), K (number of portfolios). Three shrinkage effects: L2 on return variation, L2 on sample mean toward cross-sectional average, L1 for sparsity.
- **Grid** (full, matches paper): λ₀ ∈ `np.arange(0, 0.95, 0.05)`, λ₂ ∈ `10 ** -np.arange(5, 8.25, 0.25)`.

## Conventions

- PEP 8. Use descriptive names matching paper notation (`lambda_0` for λ₀).
- Keep tree construction, pruning, evaluation as separate modules.
- Set seeds; document hyperparameters for reproducibility.
- Cross-validate every intermediate output against R ground-truth under `/scratch/.../data/raw/TreeGridSearch/...`. Use structured diff tables (see Phase D pattern in `docs/status.md`).

## Reference links

- R reference code: https://www.dropbox.com/scl/fo/6exb7jb9kctgxnosejwu4/h?rlkey=1t89bhr0pj1fwyhlvx6d9yw9c&e=1&dl=0 (read `readme_data.docx` first for architecture).
- Benchmark AP-Trees portfolios: https://www.dropbox.com/scl/fo/0c6j15c75a8va9cercl3r/h?dl=0&e=1
- Data: https://www.dropbox.com/scl/fo/z5r2qm0lsi1rnc3sop6mx/h?rlkey=zis71edhnmygr3a4ftqzeupco&e=1&dl=0
