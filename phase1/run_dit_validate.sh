#!/bin/bash
#SBATCH --job-name=lf_dit
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH --time=00:20:00
#SBATCH --output=/lustre/fs1/home/ti874297/lf/latent_free/logs/%x_%j.out
#SBATCH --error=/lustre/fs1/home/ti874297/lf/latent_free/logs/%x_%j.out
set -euo pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free/phase1
PY=/lustre/fs1/home/ti874297/lf/envs/latent_free/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
echo "host $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader||true
$PY features_dit.py
echo "done $(date -Is)"
