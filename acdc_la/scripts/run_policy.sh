#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 8 --mem=80G
#SBATCH -t 03:30:00
#SBATCH -J acdc_policy
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/policy_%j.out
# The downstream recipe (review 2026-09-12): 4-frame history policy, oracle->predicted curriculum, LATP effect consistency,
# residual BC.  Arms bc / ours / ours_noeff / univla (budget-matched codes) offline, then closed-loop for bc / ours / univla.
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free
export PYTHONPATH=/lustre/fs1/home/ti874297/lf/latent_free:/lustre/fs1/home/ti874297/lf/envs/LIBERO MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
echo n | /lustre/fs1/home/ti874297/lf/envs/libero/bin/python -u -m acdc_la.downstream.libero_policy --arms ${ARMS:-bc ours ours_noeff univla} --tokenizer ${TOK:-s1_acdc_1ep} --transport ${TRANS:-s3_latp_1ep} --rollout ${NINIT:-20} --rollout-arms ${RARMS:-bc ours univla} --out ${OUT:-libero_policy} 2>&1 | grep --line-buffered -vE "xFormers|warnings.warn|Using cache|UserWarning|Gym has been|gymnasium|migration_guide"
echo "POLICY JOB exit ${PIPESTATUS[1]} $(date)"
