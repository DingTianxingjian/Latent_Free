#!/bin/bash
#SBATCH --job-name=lf_wanT
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=02:30:00
#SBATCH --output=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out
#SBATCH --error=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out
#
# Wan temporal: does the video VAE's NATIVE 4x temporal fusion encode action
# better than differencing independent single-frame latents?  Built-in smoke
# asserts z0==standalone single-frame latent and z1!=z0 (fusion is real).
set -euo pipefail
cd /anvil/scratch/x-tding2/latent_free
PY=/anvil/scratch/x-tding2/envs/latent_free/bin/python
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export LF_RIDGE_ALPHAS="0.01,0.1,1,10,100,1000,10000"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  started $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "git commit $(git rev-parse HEAD)"
$PY exp_wan_temporal.py --ms ${MS:-1 2}
echo "finished $(date -Is)"
