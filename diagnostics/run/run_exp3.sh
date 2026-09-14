#!/bin/bash
# Optional bonus: probe UniVLA's pretrained stage-1 latent action model.
# Latents are cached to cache/exp3_z_s<stride>.npz, so a rerun only re-probes.
#SBATCH --job-name=lf_exp3
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out
#SBATCH --error=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out

set -euo pipefail
cd /anvil/scratch/x-tding2/latent_free

PY=/anvil/scratch/x-tding2/envs/latent_free/bin/python
export TORCH_HOME=/anvil/scratch/x-tding2/latent_free/cache/torch
export HF_HOME=/anvil/scratch/x-tding2/hf_cache
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=12
export PYTHONUNBUFFERED=1

STRIDES="${@:-1 4}"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  started $(date -Is)"
echo "git commit $(git rev-parse HEAD)  seed 0  strides: $STRIDES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY exp3.py --device cuda --batch 32 --strides $STRIDES
echo "finished $(date -Is)"
