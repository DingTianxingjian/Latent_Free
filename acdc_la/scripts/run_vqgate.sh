#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 8 --mem=80G
#SBATCH -t 01:30:00
#SBATCH -J acdc_vqgate
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/vqgate_%j.out
# Go/no-go gate: discrete per-cell codes of s1_acdc_1ep as the policy target (downstream/libero_vqgate.py).  Offline only.
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free
export PYTHONPATH=/lustre/fs1/home/ti874297/lf/latent_free:/lustre/fs1/home/ti874297/lf/envs/LIBERO MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
echo n | /lustre/fs1/home/ti874297/lf/envs/libero/bin/python -u -m acdc_la.downstream.libero_vqgate --arms ${ARMS:-ours vq16 vq64 vq256} --tokenizer ${TOK:-s1_acdc_1ep} --out ${OUT:-libero_vqgate} 2>&1 | grep --line-buffered -vE "xFormers|warnings.warn|Using cache|UserWarning|Gym has been|gymnasium|migration_guide"
echo "VQGATE JOB exit ${PIPESTATUS[1]} $(date)"
