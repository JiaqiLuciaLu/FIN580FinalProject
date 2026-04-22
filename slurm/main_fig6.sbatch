#!/bin/bash
#SBATCH --job-name=main_fig6         # full pipeline across all 36 (feat1, feat2) pairs
#SBATCH --array=0-35                 # 36 cross-sections = C(9, 2) of non-LME chars
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4            # AP_Pruning parallelizes across lambda0 (inner pool = 4 workers)
#SBATCH --mem-per-cpu=640M           # 2.5 GB total; smoke-test peak MaxRSS ~2.08 GB (job 3124588); ~20% headroom
#SBATCH --time=01:45:00              # observed 1h 12-14m elapsed on smoke test (job 3124588); ~30% buffer
#SBATCH --output=slurm/logs/main_fig6-%A_%a.out
#SBATCH --error=slurm/logs/main_fig6-%A_%a.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=jl6134@princeton.edu

# ---------------------------------------------------------------------------
# Runs src/code/jobs/main_table1.py for every (feat1, feat2) pair in
# C(range(2, 11), 2) == 36 cross-sections (LME is always the anchor char).
#
# $SLURM_ARRAY_TASK_ID ∈ [0, 35] indexes the sorted-tuple ordering:
#   0 -> (2, 3), 1 -> (2, 4), ..., 15 -> (4, 5) = LME_OP_Investment, ..., 35 -> (9, 10)
#
# Smoke test (3 pairs):    sbatch --array=0-2 slurm/main_fig6.sbatch
# Full sweep:              sbatch slurm/main_fig6.sbatch
# ---------------------------------------------------------------------------

module purge
module load anaconda3/2025.6
conda activate FIN580

# One BLAS thread per worker — AP_Pruning owns parallelism via multiprocessing.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /scratch/network/jl6134/COLLAB/FIN580/FIN580FinalProject

PAIR=$(python -c "
import itertools
pairs = list(itertools.combinations(range(2, 11), 2))
f1, f2 = pairs[${SLURM_ARRAY_TASK_ID}]
print(f'{f1} {f2}')
")
read F1 F2 <<< "$PAIR"

echo "=== Job ${SLURM_JOB_ID} (array ${SLURM_ARRAY_TASK_ID}) started: $(date) on $(hostname) ==="
echo "CPUs: $SLURM_CPUS_PER_TASK   feat1=$F1 feat2=$F2"

PYTHONPATH=. python -u src/code/jobs/main_table1.py \
    --workers "$SLURM_CPUS_PER_TASK" \
    --feat1 "$F1" \
    --feat2 "$F2"

echo "=== Job ${SLURM_JOB_ID} (array ${SLURM_ARRAY_TASK_ID}) finished: $(date) ==="
