#!/bin/bash
#SBATCH --job-name=lf_inc
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out
#SBATCH --error=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out
#
# Item 1b: incremental probe -- what the LAM latent adds beyond single-frame
# state.  Decides whether the "VQ costs 32%/66%" headline is transition loss or
# just redundant-state removal.  All features cached; GPU probes only.
set -euo pipefail
cd /anvil/scratch/x-tding2/latent_free
PY=/anvil/scratch/x-tding2/envs/latent_free/bin/python
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export LF_RIDGE_ALPHAS="0.01,0.1,1,10,100,1000,10000"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  started $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
echo "git commit $(git rev-parse HEAD)  seed 0"
$PY exp3_incremental.py
echo "finished $(date -Is)"
