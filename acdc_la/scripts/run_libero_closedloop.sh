#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 8 --mem=100G
#SBATCH -t 08:00:00
#SBATCH -J acdc_success
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/success_%j.out
# Closed-loop success rate on LIBERO-spatial in the `libero` venv (mujoco EGL headless).
set -u -o pipefail
cd /anvil/scratch/x-tding2/latent_free
export PYTHONPATH=/anvil/scratch/x-tding2/latent_free:/anvil/scratch/x-tding2/envs/LIBERO MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
echo n | /anvil/scratch/x-tding2/envs/libero/bin/python -u -m acdc_la.downstream.libero_closedloop --tags ${TAGS:-s1_acdc_1ep s1_mean_1ep s1_acdc} --arms univla nolatent --n-init ${NINIT:-10} 2>&1 | grep --line-buffered -vE "xFormers|warnings.warn|Using cache|UserWarning|Gym has been|gymnasium|migration_guide"
echo "SUCCESS JOB exit ${PIPESTATUS[0]} $(date)"
