#!/bin/bash
# CPU fallback for the closed-loop success rate on a login node (no GPU queue): reduced arm set, 3 inits per task.
cd /anvil/scratch/x-tding2/latent_free
export PYTHONPATH=/anvil/scratch/x-tding2/latent_free:/anvil/scratch/x-tding2/envs/LIBERO MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
echo n | /anvil/scratch/x-tding2/envs/libero/bin/python -u -m acdc_la.downstream.libero_closedloop --tags s1_acdc_1ep s3_latp_1ep s1_mean_1ep --arms nolatent T --n-init 3 --max-steps 220 --steps 1500 --out libero_closedloop_cpu 2>&1 | grep --line-buffered -vE "xFormers|warnings.warn|Using cache|UserWarning|Gym has been|gymnasium|migration_guide"
echo "CPU CLOSEDLOOP DONE $(date)"
