#!/bin/bash
#SBATCH --job-name=uniq_alive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/uniq_alive-%j.out
#SBATCH --error=slurm/logs/uniq_alive-%j.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jl6134@princeton.edu

# ---------------------------------------------------------------------------
# Per-variant 'unique alive firms' diagnostic for the K-portfolio fuzzy SDF.
# Replays the K selected portfolios' weight vectors per month, takes the
# union across them, counts firms with w >= threshold.
#
# Default: K=25, threshold=0.001, all discovered variants.
#   sbatch slurm/unique_alive_diagnostic.sbatch
# Override:
#   sbatch --export=ALL,KSEL=10,THRESH=0.0005 slurm/unique_alive_diagnostic.sbatch
# ---------------------------------------------------------------------------

module purge
module load anaconda3/2025.6
conda activate FIN580

cd /scratch/network/jl6134/COLLAB/FIN580/FIN580FinalProject

KSEL="${KSEL:-25}"
THRESH="${THRESH:-0.001}"

echo "=== Job $SLURM_JOB_ID started: $(date) on $(hostname) ==="
echo "K=$KSEL threshold=$THRESH"

PYTHONPATH=. python -u src/code/metrics/unique_alive_diagnostic.py \
    --K "$KSEL" --threshold "$THRESH"

echo "=== Job $SLURM_JOB_ID finished: $(date) ==="
