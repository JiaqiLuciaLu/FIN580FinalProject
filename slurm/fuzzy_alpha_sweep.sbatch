#!/bin/bash
#SBATCH --job-name=fuzzy_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=512M
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/fuzzy_sweep-%j.out
#SBATCH --error=slurm/logs/fuzzy_sweep-%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=jl6134@princeton.edu

# ---------------------------------------------------------------------------
# Sweep the sigmoid steepness alpha for the fuzzy AP-Tree on
# LME_OP_Investment. For each alpha, run the full pipeline (steps 2-4 +
# AP_Pruning + pickBestLambda + SDF + pickSRN) on the paper grid (19x13)
# and stash outputs under `LME_OP_Investment_a{alpha}` so they do not
# clobber each other or the alpha=10 baseline already on disk.
#
# Override alphas via:
#   sbatch --export=ALL,ALPHAS="5 20 50 100" slurm/fuzzy_alpha_sweep.sbatch
# ---------------------------------------------------------------------------

module purge
module load anaconda3/2025.6
conda activate FIN580

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /scratch/network/jl6134/COLLAB/FIN580/FIN580FinalProject

echo "=== Job $SLURM_JOB_ID started: $(date) on $(hostname) ==="
echo "CPUs: $SLURM_CPUS_PER_TASK   Mem/cpu: ${SLURM_MEM_PER_CPU:-n/a}"

# alphas to sweep — alpha=10 already on disk under _a10
ALPHAS="${ALPHAS:-5 20 50 100}"

PROC=/scratch/network/jl6134/COLLAB/FIN580/data/processed
SUBDIR=LME_OP_Investment

for ALPHA in $ALPHAS; do
    echo ""
    echo "==================== alpha=$ALPHA ===================="

    # Strip the unsuffixed dirs in case a previous iteration left them.
    rm -rf "$PROC/fuzzy_tree_portfolio_quantile/$SUBDIR"
    rm -rf "$PROC/FuzzyTreeGridSearch/$SUBDIR"
    rm -rf "$PROC/FuzzyTreeGridSearch/SDFTests/$SUBDIR"

    # If outputs for this alpha already exist (resume case), skip the run.
    DEST_GRID="$PROC/FuzzyTreeGridSearch/${SUBDIR}_a${ALPHA}"
    if [ -f "$DEST_GRID/SR_N.csv" ]; then
        echo "alpha=$ALPHA: SR_N.csv already exists at $DEST_GRID — skipping run."
        continue
    fi

    PYTHONPATH=. python -u src/code/jobs/run_fuzzy_benchmark.py \
        --grid full --alpha "$ALPHA" --workers "$SLURM_CPUS_PER_TASK" --rebuild
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "alpha=$ALPHA: runner exited with $rc — leaving outputs unsuffixed for inspection."
        continue
    fi

    # Stash this alpha's outputs.
    mv "$PROC/fuzzy_tree_portfolio_quantile/$SUBDIR"     "$PROC/fuzzy_tree_portfolio_quantile/${SUBDIR}_a${ALPHA}"
    mv "$PROC/FuzzyTreeGridSearch/$SUBDIR"               "$PROC/FuzzyTreeGridSearch/${SUBDIR}_a${ALPHA}"
    mv "$PROC/FuzzyTreeGridSearch/SDFTests/$SUBDIR"      "$PROC/FuzzyTreeGridSearch/SDFTests/${SUBDIR}_a${ALPHA}"
    echo "alpha=$ALPHA: outputs saved under *_a${ALPHA}/"
done

echo "=== Job $SLURM_JOB_ID finished: $(date) ==="
