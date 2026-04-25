#!/bin/bash
#SBATCH --job-name=fuzzy_pruned
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=512M
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/fuzzy_pruned-%j.out
#SBATCH --error=slurm/logs/fuzzy_pruned-%j.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jl6134@princeton.edu

# ---------------------------------------------------------------------------
# Fuzzy AP-Tree at α=50 with manual dead-stock pruning before the leaf-return
# weighted average. Tests whether the soft tail of below-threshold weights
# is contributing useful signal or just overfitting noise.
#
# Default: α=50, dead_threshold=0.001 (matches the dead-stocks diagnostic).
#   sbatch slurm/fuzzy_dead_pruned.sbatch
#
# Override (e.g. tighter / looser prune):
#   sbatch --export=ALL,ALPHA=50,DEAD_THRESH=0.01,SUFFIX=dt010 slurm/fuzzy_dead_pruned.sbatch
# ---------------------------------------------------------------------------

module purge
module load anaconda3/2025.6
conda activate FIN580

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /scratch/network/jl6134/COLLAB/FIN580/FIN580FinalProject

ALPHA="${ALPHA:-50}"
DEAD_THRESH="${DEAD_THRESH:-0.001}"
# Default suffix encodes both alpha and threshold so sweeps don't clash.
DEFAULT_SUFFIX="a${ALPHA}_dt$(printf '%g' "$DEAD_THRESH" | tr -d '.')"
SUFFIX="${SUFFIX:-$DEFAULT_SUFFIX}"

PROC=/scratch/network/jl6134/COLLAB/FIN580/data/processed
SUBDIR=LME_OP_Investment

echo "=== Job $SLURM_JOB_ID started: $(date) on $(hostname) ==="
echo "alpha=$ALPHA dead_threshold=$DEAD_THRESH suffix=$SUFFIX"

# Clean unsuffixed dirs in case a prior job left them behind.
rm -rf "$PROC/fuzzy_tree_portfolio_quantile/$SUBDIR"
rm -rf "$PROC/FuzzyTreeGridSearch/$SUBDIR"
rm -rf "$PROC/FuzzyTreeGridSearch/SDFTests/$SUBDIR"

# Skip if outputs for this suffix already exist.
DEST_GRID="$PROC/FuzzyTreeGridSearch/${SUBDIR}_${SUFFIX}"
if [ -f "$DEST_GRID/SR_N.csv" ]; then
    echo "$SUFFIX: SR_N.csv already exists at $DEST_GRID — skipping run."
    echo "=== Job $SLURM_JOB_ID finished: $(date) ==="
    exit 0
fi

PYTHONPATH=. python -u src/code/jobs/run_fuzzy_benchmark.py \
    --grid full --alpha "$ALPHA" --dead-threshold "$DEAD_THRESH" \
    --workers "$SLURM_CPUS_PER_TASK" --rebuild
rc=$?
if [ $rc -ne 0 ]; then
    echo "runner exited with $rc — leaving outputs unsuffixed for inspection."
    exit $rc
fi

mv "$PROC/fuzzy_tree_portfolio_quantile/$SUBDIR"   "$PROC/fuzzy_tree_portfolio_quantile/${SUBDIR}_${SUFFIX}"
mv "$PROC/FuzzyTreeGridSearch/$SUBDIR"             "$PROC/FuzzyTreeGridSearch/${SUBDIR}_${SUFFIX}"
mv "$PROC/FuzzyTreeGridSearch/SDFTests/$SUBDIR"    "$PROC/FuzzyTreeGridSearch/SDFTests/${SUBDIR}_${SUFFIX}"
echo "outputs saved under *_${SUFFIX}/"

echo "=== Job $SLURM_JOB_ID finished: $(date) ==="
