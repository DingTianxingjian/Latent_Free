#!/bin/bash
#SBATCH --job-name=lf_dit
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out
#SBATCH --error=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out
set -euo pipefail
cd /anvil/scratch/x-tding2/latent_free/phase1
PY=/anvil/scratch/x-tding2/envs/latent_free/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
echo "host $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader||true
$PY features_dit.py
echo "done $(date -Is)"
