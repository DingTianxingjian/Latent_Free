#!/bin/bash
# GPU stage only (DINOv2 feature extraction) on the gpu-debug queue.
# gpu-debug: 30 min wall, 1 running job -- so this is resumable: conditions that
# are already cached are skipped, and a condition that will not fit in the
# remaining budget is deferred.  Just resubmit until it prints "pending: none".
#SBATCH --job-name=lf_feat
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  started $(date -Is)"
echo "git commit $(git rev-parse HEAD)  seed 0"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY data.py
# leave ~4 min of headroom inside the 30 min wall clock
$PY features.py --examples --budget-min 25 \
    --conditions clean patch patch_static border border_static

echo "finished $(date -Is)"
ls -la cache/feats_*.npy 2>/dev/null || true
