#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 8 --mem=60G
#SBATCH -t 02:30:00
#SBATCH -J acdc_acc
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/acc_%j.out
# THE ACCURACY TABLE, one job: (1) LIBERO offline latent_acc + action token accuracy, (2) closed-loop success rate.
# Short walltime on purpose (backfill).  Submit with --account/--partition on the command line.
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free
PY=/lustre/fs1/home/ti874297/lf/envs/latent_free/bin/python
Q='xFormers|warnings.warn|Using cache|UserWarning|Gym has been|gymnasium|migration_guide'
if [ "${SKIP_OFFLINE:-0}" != 1 ]; then
$PY -u -m acdc_la.downstream.libero_offline --tags ${TAGS:-s1_acdc_1ep s1_mean_1ep s1_acdc} --steps 2000 --univla-arm ${UARM:-official_stage2} --out ${OUT:-libero_offline} 2>&1 | grep --line-buffered -vE "$Q"
echo "OFFLINE exit ${PIPESTATUS[0]} $(date)"
fi
export PYTHONPATH=/lustre/fs1/home/ti874297/lf/latent_free:/lustre/fs1/home/ti874297/lf/envs/LIBERO MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
echo n | /lustre/fs1/home/ti874297/lf/envs/libero/bin/python -u -m acdc_la.downstream.libero_closedloop --tags ${TAGS:-s1_acdc_1ep s1_mean_1ep s1_acdc} --arms univla nolatent T --n-init ${NINIT:-5} --max-steps ${MAXSTEPS:-220} --univla-arm ${UARM:-official_stage2} --steps 2000 --out ${OUT:-libero}_closedloop 2>&1 | grep --line-buffered -vE "$Q"
echo "CLOSEDLOOP exit ${PIPESTATUS[1]} $(date)"
echo "ACCURACY JOB DONE $(date)"
