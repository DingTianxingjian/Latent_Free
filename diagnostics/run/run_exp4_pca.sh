#!/bin/bash
# Exp 4 -- patch-token features + the single/diff/pair matrix on clean frames.
#
# Step 1 is a parity check of the GPU probes against the sklearn probes that
# produced exp 1, on the real cached CLS features.  If it fails the job aborts,
# because every exp-4 number comes out of the GPU path.
#SBATCH --job-name=lf_exp4pca
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fs1/home/ti874297/lf/latent_free/logs/%x_%j.out
#SBATCH --error=/lustre/fs1/home/ti874297/lf/latent_free/logs/%x_%j.out

set -euo pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free

PY=/lustre/fs1/home/ti874297/lf/envs/latent_free/bin/python
export TORCH_HOME=/lustre/fs1/home/ti874297/lf/latent_free/cache/torch
export HF_HOME=/lustre/fs1/home/ti874297/lf/hf_cache
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=12
export PYTHONUNBUFFERED=1

echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  started $(date -Is)"
echo "git commit $(git rev-parse HEAD)  seed 0"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo; echo "=== width-matched PCA controls: 12288-d patch grid -> 768-d and 512-d ==="
:

:

$PY exp4_patch.py --condition clean --device cuda --spaces pg4pca768 pg4pca512 --budget-min 22

echo; echo "finished $(date -Is)"
