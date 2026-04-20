#!/bin/bash
#SBATCH --job-name=main_tail
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00              # pickSRN <5 min + 2-char tree ~16 min; 1h is plenty
#SBATCH --output=slurm/logs/main_tail-%j.out
#SBATCH --error=slurm/logs/main_tail-%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=jl6134@princeton.edu

# ---------------------------------------------------------------------------
# Re-run the tail of main.py that crashed (pickSRN + 2-char tree).
# Requires prior main.py outputs in PY_TREE_GRID_DIR and PY_TREE_PORT_DIR.
# ---------------------------------------------------------------------------

module purge
module load anaconda3/2025.6
conda activate FIN580

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /scratch/network/jl6134/COLLAB/FIN580/FIN580FinalProject

echo "=== Job $SLURM_JOB_ID started: $(date) on $(hostname) ==="
echo "CPUs: $SLURM_CPUS_PER_TASK   Mem/cpu: ${SLURM_MEM_PER_CPU:-n/a}M"

PYTHONPATH=. python -u src/code/run_main_tail.py

echo "=== Job $SLURM_JOB_ID finished: $(date) ==="
