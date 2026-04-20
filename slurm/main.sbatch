#!/bin/bash
#SBATCH --job-name=main_full         # create a short name for your job
#SBATCH --nodes=1                    # node count
#SBATCH --ntasks=1                   # total number of tasks across all nodes
#SBATCH --cpus-per-task=1            # RunParallel=False → pipeline is single-threaded
#SBATCH --mem-per-cpu=4G             # previous run peaked at ~1 GB; 4 GB gives 4x headroom
#SBATCH --time=05:00:00              # ~3.5 h estimated; 5 h leaves buffer
#SBATCH --output=slurm/logs/main-%j.out
#SBATCH --error=slurm/logs/main-%j.err
#SBATCH --mail-type=begin,end,fail   # receive email notifications
#SBATCH --mail-user=jl6134@princeton.edu

# ---------------------------------------------------------------------------
# Full-pipeline clean-data run: src/code/main.py
#
# Pipeline (single cross-section, LME_OP_Investment; 3x3 demo λ grid):
#   1. Step 1  create_yearly_chunks(add_noise=False) from raw CRSP panels
#   2. Step 2  create_tree_portfolio (81 feature permutations)
#   3. Step 3  combinetrees (dedup + subtract rf)
#   4. Step 4  filter monoculture
#   5. gen_triple_sort_32 + gen_triple_sort_64 (base portfolios ts32, ts64)
#   6. AP_Pruning on tree + ts32 + ts64 (9 grid cells each)
#   7. pickBestLambda: tree @ K=10, ts32 @ K=10 & 32, ts64 @ K=10 & 64
#   8. SDF_regression on tree + ts32 + ts64  → Paper Table 1 row
#   9. pickSRN on tree (K=5..50)              → Figure 10 curve
#  10. 2-char tree + combine_2char            → Figure 1b inputs
#
# Runtime estimate (serial, RunParallel=False):
#     Step 1   ~5-10 min
#     Step 2   ~90 min (81 tree permutations)
#     ts32/64  ~30 + 45 min
#     2-char   ~20 min
#     rest     ~10 min
#     TOTAL    ~3.5 h
#
# Resource sizing (based on seff of previous main_simplified run):
#     CPU efficiency  12.4% on 8 cores  → 1 core is sufficient
#     Memory peak     1 GB / 32 GB      → 4 GB more than enough
#
# Prereqs:
#   • Conda env FIN580 (Python 3.11 + numpy/pandas/scipy/scikit-learn/statsmodels).
#   • Raw CRSP panels:  raw/ap_char/characteristics/{RET,LME,OP,Investment}.csv
#   • Factor files:     raw/data/factor/{rf_factor,tradable_factors}.csv
#   • Log directory:    mkdir -p slurm/logs   (once, before first submission)
#
# Submit:
#   sbatch slurm/main.sbatch
# ---------------------------------------------------------------------------

module purge
module load anaconda3/2025.6
conda activate FIN580

# Single-threaded BLAS — the pipeline is serial; no benefit to extra threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /scratch/network/jl6134/COLLAB/FIN580/FIN580FinalProject

echo "=== Job $SLURM_JOB_ID started: $(date) on $(hostname) ==="
echo "CPUs: $SLURM_CPUS_PER_TASK   Mem/cpu: ${SLURM_MEM_PER_CPU:-n/a}M"
echo "Working dir: $(pwd)"

PYTHONPATH=. python -u src/code/main.py

echo "=== Job $SLURM_JOB_ID finished: $(date) ==="
