#!/bin/bash
#SBATCH --job-name=fuzzy_dt_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=512M
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/fuzzy_dt_sweep-%j.out
#SBATCH --error=slurm/logs/fuzzy_dt_sweep-%j.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jl6134@princeton.edu

# ---------------------------------------------------------------------------
# Sweep dead-stock pruning thresholds at fuzzy α=50 and append pruning-focused
# rows to the K=25 comparison CSV. Default sweep: 1%, 2%, 5%.
#
# Override:
#   sbatch --export=ALL,DTS="0.01 0.025 0.05 0.1" slurm/fuzzy_dt_sweep.sbatch
# ---------------------------------------------------------------------------

module purge
module load anaconda3/2025.6
conda activate FIN580

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /scratch/network/jl6134/COLLAB/FIN580/FIN580FinalProject

DTS="${DTS:-0.01 0.02 0.05}"
ALPHA=50
PROC=/scratch/network/jl6134/COLLAB/FIN580/data/processed
SUBDIR=LME_OP_Investment

echo "=== Job $SLURM_JOB_ID started: $(date) on $(hostname) ==="
echo "alpha=$ALPHA  thresholds=$DTS"

# Phase 1: run the pipeline for each threshold (skip if already done).
for DT in $DTS; do
    SUFFIX_ENC=$(printf '%g' "$DT" | tr -d '.')
    SUFFIX="a${ALPHA}_dt${SUFFIX_ENC}"
    DEST="$PROC/FuzzyTreeGridSearch/${SUBDIR}_${SUFFIX}"

    echo ""
    echo "==================== dt=$DT  suffix=$SUFFIX ===================="

    if [ -f "$DEST/SR_N.csv" ]; then
        echo "$SUFFIX already complete — skipping pipeline."
    else
        rm -rf "$PROC/fuzzy_tree_portfolio_quantile/$SUBDIR"
        rm -rf "$PROC/FuzzyTreeGridSearch/$SUBDIR"
        rm -rf "$PROC/FuzzyTreeGridSearch/SDFTests/$SUBDIR"

        PYTHONPATH=. python -u src/code/jobs/run_fuzzy_benchmark.py \
            --grid full --alpha "$ALPHA" --dead-threshold "$DT" \
            --workers "$SLURM_CPUS_PER_TASK" --rebuild
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "dt=$DT: runner exited with $rc — skipping move and stats."
            continue
        fi
        mv "$PROC/fuzzy_tree_portfolio_quantile/$SUBDIR"   "$PROC/fuzzy_tree_portfolio_quantile/${SUBDIR}_${SUFFIX}"
        mv "$PROC/FuzzyTreeGridSearch/$SUBDIR"             "$DEST"
        mv "$PROC/FuzzyTreeGridSearch/SDFTests/$SUBDIR"    "$PROC/FuzzyTreeGridSearch/SDFTests/${SUBDIR}_${SUFFIX}"
        echo "outputs saved under *_${SUFFIX}/"
    fi
done

# Phase 2: compute avg_alive_per_portfolio per threshold, append rows to the
# focused-comparison CSV. Re-runnable: existing rows for these dt values get
# replaced before append.
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

new_rows = []
for dt in DTS:
    enc = ("%g" % dt).replace(".", "")
    label = f"fuzzy α=50, dt={dt*100:g}% prune"
    grid_dir = os.path.join(utils.PY_FUZZY_TREE_GRID_DIR, f"{subdir}_a50_dt{enc}")
    if not os.path.isfile(os.path.join(grid_dir, "SR_N.csv")):
        print(f"[skip] missing SR_N.csv at {grid_dir}")
        continue

    t0 = time.time()
    s = _stats_one_variant(
        grid_dir=grid_dir, alpha_value=50.0, K=25, threshold=dt,
        chunks_dir=chunks_dir, feats=feats, y_min=1964, y_max=2016,
    )
    print(f"[{label}] alive/port={s['avg_alive_per_portfolio']:.1f}  "
          f"unique/mo={s['avg_unique_alive_per_month']:.1f}  "
          f"compute={time.time()-t0:.1f}s")

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
    print(f"\nwrote {focused}")
    print(out.to_string(index=False))
else:
    print("no rows to append")
PYEOF

echo "=== Job $SLURM_JOB_ID finished: $(date) ==="
