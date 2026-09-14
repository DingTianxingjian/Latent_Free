#!/bin/bash
#SBATCH --job-name=lf_dbg
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out
#SBATCH --error=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out
set -euo pipefail
cd /anvil/scratch/x-tding2/latent_free/phase1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
/anvil/scratch/x-tding2/envs/latent_free/bin/python debug_dit_vae.py
