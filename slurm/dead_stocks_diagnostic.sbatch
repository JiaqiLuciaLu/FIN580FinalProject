#!/bin/bash
#SBATCH --job-name=dead_stocks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/dead_stocks-%j.out
#SBATCH --error=slurm/logs/dead_stocks-%j.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jl6134@princeton.edu

# ---------------------------------------------------------------------------
# Dead-stocks diagnostic for the fuzzy AP-Tree at LME_OP_Investment.
# Default: alpha=50, K=25, threshold=0.001 (0.1%). Override via:
#   sbatch --export=ALL,ALPHA=100,KSEL=22,THRESH=0.0005 slurm/dead_stocks_diagnostic.sbatch
# ---------------------------------------------------------------------------

module purge
module load anaconda3/2025.6
conda activate FIN580

cd /scratch/network/jl6134/COLLAB/FIN580/FIN580FinalProject

ALPHA="${ALPHA:-50}"
KSEL="${KSEL:-25}"
THRESH="${THRESH:-0.001}"

echo "=== Job $SLURM_JOB_ID started: $(date) on $(hostname) ==="
echo "alpha=$ALPHA K=$KSEL threshold=$THRESH"

PYTHONPATH=. python -u src/code/metrics/dead_stocks_diagnostic.py \
    --alpha "$ALPHA" --K "$KSEL" --threshold "$THRESH"

echo "=== Job $SLURM_JOB_ID finished: $(date) ==="
