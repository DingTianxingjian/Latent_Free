#!/bin/bash
# Exp 4 -- patch-token features + the single/diff/pair matrix on clean frames.
#
# Step 1 is a parity check of the GPU probes against the sklearn probes that
# produced exp 1, on the real cached CLS features.  If it fails the job aborts,
# because every exp-4 number comes out of the GPU path.
#SBATCH --job-name=lf_exp4
#SBATCH --account=cis250950-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=00:40:00
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
export LF_OUT_SUFFIX=_a100   # distinct output: the ai job may run concurrently

echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  started $(date -Is)"
echo "git commit $(git rev-parse HEAD)  seed 0"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo; echo "=== 1/3  GPU probes vs sklearn parity, on the real exp-1 features ==="
$PY probes_torch.py --verify --device cuda

echo; echo "=== 2/3  patch-token features, clean ==="
$PY features_patch.py --conditions clean --device cuda --batch 64

echo; echo "=== 3/3  cls / pmean / pg4  x  single / diff / pair ==="
$PY exp4_patch.py --condition clean --device cuda --feats-cpu --budget-min "${EXP4_BUDGET_MIN:-25}"

echo; echo "finished $(date -Is)"
