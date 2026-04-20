#!/bin/bash
#SBATCH --job-name=main_simplified   # create a short name for your job
#SBATCH --nodes=1                    # node count
#SBATCH --ntasks=1                   # total number of tasks across all nodes
#SBATCH --cpus-per-task=8            # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G             # memory per cpu-core (4G is default)
#SBATCH --time=03:00:00              # total run time limit (HH:MM:SS) — Step 1 adds ~5-15min
#SBATCH --output=slurm/logs/main_simplified-%j.out
#SBATCH --error=slurm/logs/main_simplified-%j.err
#SBATCH --mail-type=begin,end,fail   # receive email notifications
#SBATCH --mail-user=jl6134@princeton.edu

# ---------------------------------------------------------------------------
# Faithful-replication smoke test: runs src/code/main_simplified.py end-to-end.
#
# Pipeline (single cross-section, LME_OP_Investment; one λ point 0.15/1e-8):
#   1. Tree portfolio Step 2 (create_tree_portfolio)   — dominant cost
#   2. Tree portfolio Step 3 (combinetrees, subtract rf)
#   3. Tree portfolio Step 4 (filter monoculture)
#   4. AP_Pruning (2 LARS fits: cv_3 fold + full)
#   5. pickBestLambda at K=10
#   6. SDF_regression (FF3 / FF5 / XSF / FF11 alpha)
# Prints SR tuple + regression result table to stdout.
#
# Prereqs:
#   • Conda env FIN580 exists (Python 3.11 + numpy/pandas/scipy/scikit-learn/statsmodels).
#   • Input chunks present:   raw/data/data_chunk_files_quantile/LME_OP_Investment/y{1964..2016}.csv
#   • Factor files present:   raw/data/factor/{rf_factor,tradable_factors}.csv
#   • Log directory:          mkdir -p slurm/logs   (once, before first submission)
#
# Submit:
#   sbatch slurm/main_simplified.sbatch
# ---------------------------------------------------------------------------

module purge
module load anaconda3/2025.6
conda activate FIN580

# Keep BLAS libraries single-threaded so they do not oversubscribe the
# multiprocessing.Pool workers (main_simplified.py uses pralleln=6).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /scratch/network/jl6134/COLLAB/FIN580/FIN580FinalProject

echo "=== Job $SLURM_JOB_ID started: $(date) on $(hostname) ==="
echo "CPUs: $SLURM_CPUS_PER_TASK   Mem/cpu: ${SLURM_MEM_PER_CPU:-n/a}M"
echo "Working dir: $(pwd)"

PYTHONPATH=. python -u src/code/main_simplified.py

echo "=== Job $SLURM_JOB_ID finished: $(date) ==="
