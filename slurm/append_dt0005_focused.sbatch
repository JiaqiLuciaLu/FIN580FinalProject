#!/bin/bash
#SBATCH --job-name=append_dt5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/append_dt5-%j.out
#SBATCH --error=slurm/logs/append_dt5-%j.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jl6134@princeton.edu

# Compute avg_alive_per_portfolio for the dt=0.5% variant at threshold 0.005
# (matches the dt cutoff) and append a row to the pruning-focused comparison.

module purge
module load anaconda3/2025.6
conda activate FIN580
cd /scratch/network/jl6134/COLLAB/FIN580/FIN580FinalProject

PYTHONPATH=. python -u - <<'EOF'
import os, time, pandas as pd
from src.code import utils
from src.code.metrics.unique_alive_diagnostic import _stats_one_variant

feats = ["LME", "OP", "Investment"]
subdir = "LME_OP_Investment"
chunks_dir = os.path.join(utils.PY_DATA_CHUNK_DIR, subdir)
grid_dir = os.path.join(utils.PY_FUZZY_TREE_GRID_DIR, "LME_OP_Investment_a50_dt0005")

t0 = time.time()
stats = _stats_one_variant(
    grid_dir=grid_dir, alpha_value=50.0, K=25, threshold=0.005,
    chunks_dir=chunks_dir, feats=feats, y_min=1964, y_max=2016,
)
print(f"compute time: {time.time()-t0:.1f}s")
print(f"avg_alive_per_portfolio (w >= 0.005) : {stats['avg_alive_per_portfolio']:.1f}")
print(f"avg unique alive / month             : {stats['avg_unique_alive_per_month']:.1f}")
print(f"depth mix (d0..d4)                   : "
      f"{stats['depth_count_d0']} {stats['depth_count_d1']} {stats['depth_count_d2']} "
      f"{stats['depth_count_d3']} {stats['depth_count_d4']}")

df = pd.read_csv(os.path.join(grid_dir, "SR_N.csv"))
col = 25 - 5  # K=25, mink=5
train_sr, valid_sr, test_sr = df.iloc[0, col], df.iloc[1, col], df.iloc[2, col]
print(f"K=25 SR train/valid/test : {train_sr:.4f} / {valid_sr:.4f} / {test_sr:.4f}")

# Append to the focused comparison CSV.
focused = ("/scratch/network/jl6134/COLLAB/FIN580/FIN580FinalProject/plots/abalation/"
           "abalation1_LME_OP_Investment_pruning_focused_K25.csv")
existing = pd.read_csv(focused)
new_row = pd.DataFrame([{
    "variant": "fuzzy α=50, dt=0.5% prune",
    "train_SR": train_sr, "valid_SR": valid_sr, "test_SR": test_sr,
    "total_firms_per_port": stats["avg_alive_per_portfolio"],
    "meaningful_firms_per_port": stats["avg_alive_per_portfolio"],
    "comment": "dead tail (w<0.005) zeroed before VW return",
}])

# Drop any prior dt=0.5% row in case this is a rerun.
existing = existing[~existing["variant"].str.contains("dt=0.5%", regex=False)]
out = pd.concat([existing, new_row], ignore_index=True)
out.to_csv(focused, index=False)
print(f"\nwrote {focused}\n")
print(out.to_string(index=False))
EOF
