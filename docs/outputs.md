# Generated outputs

All outputs live under `/scratch/network/jl6134/COLLAB/FIN580/data/processed/` (sibling of `raw/`, not in git). Root `$OUT = data/processed`.

| Artifact | Path | Produced by |
|---|---|---|
| Clean yearly chunks | `$OUT/data_chunk_files_quantile/LME_OP_Investment/y{1964..2016}.csv` | `main.py` Step 1 |
| Tree per-perm portfolios (81×7) | `$OUT/tree_portfolio_quantile/LME_OP_Investment/{perm_id}{ret,LME_min,LME_max,OP_min,OP_max,Investment_min,Investment_max}.csv` | `main.py` Step 2 |
| Tree combined + filtered | `$OUT/tree_portfolio_quantile/LME_OP_Investment/level_all_*.csv` and `level_all_*_filtered.csv` | `main.py` Steps 3–4 |
| Triple-sort 32 / 64 | `$OUT/ts_portfolio/LME_OP_Investment/excess_ports.csv`, `$OUT/ts64_portfolio/LME_OP_Investment/excess_ports.csv` | `main.py` |
| AP pruning grid (3×3 demo) | `$OUT/{TreeGridSearch,TSGridSearch,TS64GridSearch}/LME_OP_Investment/results_{cv_3,full}_l0_{1..3}_l2_{1..3}.csv` | `main.py` AP_Pruning |
| Selected portfolios + weights (K=10, 32, 64) | `$OUT/{TreeGridSearch,TSGridSearch,TS64GridSearch}/LME_OP_Investment/Selected_Ports_{K}.csv`, `Selected_Ports_Weights_{K}.csv` | `main.py` pickBestLambda |
| Train / valid / test SR matrices per K | `$OUT/{…GridSearch}/LME_OP_Investment/{train,valid,test}_SR_{K}.csv` | `main.py` pickBestLambda |
| SDF time-series α / SE / t / p (FF3, FF5, XSF, FF11) | `$OUT/{TreeGridSearch,TSGridSearch,TS64GridSearch}/SDFTests/LME_OP_Investment/TimeSeriesAlpha.csv` | `main.py` SDF_regression |
| SR-vs-K curve (Figure 10 input) | `$OUT/TreeGridSearch/LME_OP_Investment/SR_N.csv` | `run_main_tail.py` pickSRN |
| 2-char tree (Figure 1b input) | `$OUT/tree_portfolio_quantile/LME_OP/level_all_{LME,OP}_{min,max}.csv` + 80 per-perm files | `run_main_tail.py` 2-char tree |
| Noisy-run pipeline-validation plot | `$OUT/plots/Noisy_replication.{csv,png}` | ad-hoc post-main_simplified |

SLURM logs: `slurm/logs/main-{jobid}.{out,err}`, `slurm/logs/main_tail-{jobid}.{out,err}`, `slurm/logs/main_simplified-{jobid}.{out,err}`.

Runs producing the current state: **main.py job 3117381** (1h35m, all stages except pickSRN and 2-char tree) + **run_main_tail.py job 3117843** (18 min, pickSRN + 2-char tree).
