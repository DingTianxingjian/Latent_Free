#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 4 --mem=40G
#SBATCH -t 00:20:00
#SBATCH -J acdc_frames
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/frames_%j.out
#SBATCH --exclude=h001,h003,h004,h006,h009,h017
# Render LIBERO initial-state frames for the paper's environment figure.  Needs a GPU for EGL.
set -u -o pipefail
cd /anvil/scratch/x-tding2/latent_free
export PYTHONPATH=/anvil/scratch/x-tding2/latent_free:/anvil/scratch/x-tding2/latent_free/third_party/UniVLA:/anvil/scratch/x-tding2/envs/LIBERO
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
/anvil/scratch/x-tding2/envs/univla_vla/bin/python -m acdc_la.figures.grab_libero_frames --tasks ${TASKS:-3} --res ${RES:-512}
echo "FRAMES exit $? $(date)"
