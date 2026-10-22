#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 8 --mem=100G
#SBATCH -t 08:00:00
#SBATCH -J acdc_success
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/success_%j.out
# Closed-loop success rate on LIBERO-spatial in the `libero` venv (mujoco EGL headless).
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free
export PYTHONPATH=/lustre/fs1/home/ti874297/lf/latent_free:/lustre/fs1/home/ti874297/lf/envs/LIBERO MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
echo n | /lustre/fs1/home/ti874297/lf/envs/libero/bin/python -u -m acdc_la.downstream.libero_closedloop --tags ${TAGS:-s1_acdc_1ep s1_mean_1ep s1_acdc} --arms univla nolatent --n-init ${NINIT:-10} 2>&1 | grep --line-buffered -vE "xFormers|warnings.warn|Using cache|UserWarning|Gym has been|gymnasium|migration_guide"
echo "SUCCESS JOB exit ${PIPESTATUS[0]} $(date)"
