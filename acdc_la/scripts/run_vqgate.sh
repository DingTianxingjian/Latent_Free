#!/bin/bash
#SBATCH --account=cis250950-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 8 --mem=80G
#SBATCH -t 01:30:00
#SBATCH -J acdc_vqgate
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/vqgate_%j.out
# Go/no-go gate: discrete per-cell codes of s1_acdc_1ep as the policy target (downstream/libero_vqgate.py).  Offline only.
set -u -o pipefail
cd /anvil/scratch/x-tding2/latent_free
export PYTHONPATH=/anvil/scratch/x-tding2/latent_free:/anvil/scratch/x-tding2/envs/LIBERO MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
echo n | /anvil/scratch/x-tding2/envs/libero/bin/python -u -m acdc_la.downstream.libero_vqgate --arms ${ARMS:-ours vq16 vq64 vq256} --tokenizer ${TOK:-s1_acdc_1ep} --out ${OUT:-libero_vqgate} 2>&1 | grep --line-buffered -vE "xFormers|warnings.warn|Using cache|UserWarning|Gym has been|gymnasium|migration_guide"
echo "VQGATE JOB exit ${PIPESTATUS[1]} $(date)"
