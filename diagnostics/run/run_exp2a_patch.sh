#!/bin/bash
#SBATCH --job-name=lf_2a
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out
#SBATCH --error=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out
#
# Item 2: Exp 2A in patch space + per-condition null.  First extracts patch
# features for the 4 distractor conditions (clean already cached), then runs the
# within-condition probe set and the top-k/random-k/bottom-k null on the g4 grid.
set -euo pipefail
cd /anvil/scratch/x-tding2/latent_free
PY=/anvil/scratch/x-tding2/envs/latent_free/bin/python
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export LF_RIDGE_ALPHAS="0.01,0.1,1,10,100,1000,10000"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  started $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
echo "git commit $(git rev-parse HEAD)  seed 0"

echo "=== 1/2  extract patch features for the 4 distractor conditions ==="
$PY features_patch.py --conditions patch patch_static border border_static --batch 64

echo "=== 2/2  Exp 2A probes + per-condition null ==="
$PY exp2a_patch.py
echo "finished $(date -Is)"
