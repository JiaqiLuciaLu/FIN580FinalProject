#!/bin/bash
#SBATCH --job-name=fuzzy_bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=512M
#SBATCH --time=01:45:00
#SBATCH --output=slurm/logs/fuzzy_bench-%j.out
#SBATCH --error=slurm/logs/fuzzy_bench-%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=jl6134@princeton.edu

# ---------------------------------------------------------------------------
# Fuzzy (soft-split) AP-Tree benchmark on LME_OP_Investment.
# Runs steps 2-4 of the fuzzy pipeline + AP_Pruning + SR metrics with a
# small (3x2) lambda grid by default. Switch to --grid full for the paper
# 19x13 grid once the pilot looks reasonable.
#
# Reuses yearly quantile-normalized chunks from the hard-split pipeline —
# no step 1 needed.
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
echo "Working dir: $(pwd)"

# Override with --grid full --alpha <x> on the command line when submitting
# via: sbatch --export=ALL,GRID=full,ALPHA=10 slurm/fuzzy_benchmark.sbatch
GRID="${GRID:-simple}"
ALPHA="${ALPHA:-10}"

PYTHONPATH=. python -u src/code/jobs/run_fuzzy_benchmark.py \
    --grid "$GRID" --alpha "$ALPHA" --workers "$SLURM_CPUS_PER_TASK"

echo "=== Job $SLURM_JOB_ID finished: $(date) ==="
