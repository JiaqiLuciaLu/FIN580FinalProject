#!/bin/bash
#SBATCH --job-name=append_dts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/append_dts-%j.out
#SBATCH --error=slurm/logs/append_dts-%j.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jl6134@princeton.edu

# Phase-2 fix-up: compute avg_alive_per_portfolio for the 1%/2%/5% pruned
# variants and append rows to the focused-comparison CSV. Re-runnable.

module purge
module load anaconda3/2025.6
conda activate FIN580
cd /scratch/network/jl6134/COLLAB/FIN580/FIN580FinalProject

DTS="${DTS:-0.01 0.02 0.05}"
export DTS_ENV="$DTS"

PYTHONPATH=. python -u - <<'PYEOF'
import os, time
import pandas as pd
from src.code import utils
from src.code.metrics.unique_alive_diagnostic import _stats_one_variant

feats = ["LME", "OP", "Investment"]
subdir = "LME_OP_Investment"
chunks_dir = os.path.join(utils.PY_DATA_CHUNK_DIR, subdir)
focused = ("/scratch/network/jl6134/COLLAB/FIN580/FIN580FinalProject/plots/"
           "abalation/abalation1_LME_OP_Investment_pruning_focused_K25.csv")

DTS = [float(x) for x in os.environ["DTS_ENV"].split()]
print("DTS:", DTS, flush=True)

new_rows = []
for dt in DTS:
    enc = ("%g" % dt).replace(".", "")
    label = f"fuzzy α=50, dt={dt*100:g}% prune"
    grid_dir = os.path.join(utils.PY_FUZZY_TREE_GRID_DIR, f"{subdir}_a50_dt{enc}")
    if not os.path.isfile(os.path.join(grid_dir, "SR_N.csv")):
        print(f"[skip] missing SR_N.csv at {grid_dir}", flush=True)
        continue

    t0 = time.time()
    s = _stats_one_variant(
        grid_dir=grid_dir, alpha_value=50.0, K=25, threshold=dt,
        chunks_dir=chunks_dir, feats=feats, y_min=1964, y_max=2016,
    )
    print(f"[{label}] alive/port={s['avg_alive_per_portfolio']:.1f}  "
          f"unique/mo={s['avg_unique_alive_per_month']:.1f}  "
          f"compute={time.time()-t0:.1f}s", flush=True)

    df = pd.read_csv(os.path.join(grid_dir, "SR_N.csv"))
    col = 25 - 5
    new_rows.append({
        "variant": label,
        "train_SR": df.iloc[0, col],
        "valid_SR": df.iloc[1, col],
        "test_SR":  df.iloc[2, col],
        "total_firms_per_port": s["avg_alive_per_portfolio"],
        "meaningful_firms_per_port": s["avg_alive_per_portfolio"],
        "comment": f"dead tail (w<{dt}) zeroed before VW return",
    })

if new_rows:
    existing = pd.read_csv(focused)
    new_labels = {r["variant"] for r in new_rows}
    existing = existing[~existing["variant"].isin(new_labels)]
    out = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    out.to_csv(focused, index=False)
    print(f"\nwrote {focused}\n")
    print(out.to_string(index=False))
else:
    print("no rows to append")
PYEOF
