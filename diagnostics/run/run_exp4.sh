#!/bin/bash
# Exp 4 -- patch-token features + the single/diff/pair matrix on clean frames.
#
# Step 1 is a parity check of the GPU probes against the sklearn probes that
# produced exp 1, on the real cached CLS features.  If it fails the job aborts,
# because every exp-4 number comes out of the GPU path.
#SBATCH --job-name=lf_exp4
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH --time=00:40:00
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

echo; echo "=== 1/3  GPU probes vs sklearn parity, on the real exp-1 features ==="
$PY probes_torch.py --verify --device cuda

echo; echo "=== 2/3  patch-token features, clean ==="
$PY features_patch.py --conditions clean --device cuda --batch 64

echo; echo "=== 3/3  cls / pmean / pg4  x  single / diff / pair ==="
$PY exp4_patch.py --condition clean --device cuda --budget-min "${EXP4_BUDGET_MIN:-25}"

echo; echo "finished $(date -Is)"
