#!/bin/bash
#SBATCH --job-name=main_table1       # full-grid baseline for paper Table 1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16           # AP_Pruning parallelizes across lambda0
#SBATCH --mem-per-cpu=4G             # prior run peaked ~1 GB/cpu; 4 GB headroom
#SBATCH --time=00:10:00              # prior run completed in ~2 min on cached portfolios; 10 min buffer
#SBATCH --output=slurm/logs/main_table1-%j.out
#SBATCH --error=slurm/logs/main_table1-%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=jl6134@princeton.edu

# ---------------------------------------------------------------------------
# Full-grid baseline run for paper Table 1 (cross-section LME_OP_Investment).
#
# Pipeline (see src/code/jobs/main_table1.py docstring for the full list of
# differences vs src/code/main.py):
#   • λ₀ × λ₂ grid expanded from 3×3 demo to paper's 19×13 (= 247 cells).
#   • AP_Pruning runs with RunParallel=True, workers = SLURM_CPUS_PER_TASK,
#     runFullCV=True on tree + TS32 + TS64.
#   • pickBestLambda extended to K=40 on the tree.
#   • SDF_regression extended to K=40 tree, and fixed to use K=32/64 for TS.
#   • Portfolio construction (chunks, tree, TS32/64, 2-char tree) is skipped
#     when outputs exist on disk — rerunnable cheaply.
#
# Does NOT include XS R²_adj (small local computation; run separately).
#
# Runtime estimate:
#   • AP_Pruning tree (full CV + 4 CV folds, 247 cells, parallel × 16 cpus)  ~45 min
#   • AP_Pruning TS32 / TS64                                                  ~15 min each
#   • pickBestLambda × 5, SDF_regression × 4, pickSRN                         ~10 min
#   • TOTAL wall clock ~1.5 h (6 h limit is ample)
#
# Submit:
#   sbatch slurm/main_table1.sbatch
# ---------------------------------------------------------------------------

module purge
module load anaconda3/2025.6
conda activate FIN580

# One BLAS thread per process — AP_Pruning owns parallelism via multiprocessing.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /scratch/network/jl6134/COLLAB/FIN580/FIN580FinalProject

echo "=== Job $SLURM_JOB_ID started: $(date) on $(hostname) ==="
echo "CPUs: $SLURM_CPUS_PER_TASK   Mem/cpu: ${SLURM_MEM_PER_CPU:-n/a}M"
echo "Working dir: $(pwd)"

PYTHONPATH=. python -u src/code/jobs/main_table1.py --workers "$SLURM_CPUS_PER_TASK"

echo "=== Job $SLURM_JOB_ID finished: $(date) ==="
