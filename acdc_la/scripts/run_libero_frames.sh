#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 4 --mem=40G
#SBATCH -t 00:20:00
#SBATCH -J acdc_frames
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/frames_%j.out
# Render LIBERO initial-state frames for the paper's environment figure.  Needs a GPU for EGL.
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free
export PYTHONPATH=/lustre/fs1/home/ti874297/lf/latent_free:/lustre/fs1/home/ti874297/lf/latent_free/third_party/UniVLA:/lustre/fs1/home/ti874297/lf/envs/LIBERO
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
/lustre/fs1/home/ti874297/lf/envs/univla_vla/bin/python -m acdc_la.figures.grab_libero_frames --tasks ${TASKS:-3} --res ${RES:-512}
echo "FRAMES exit $? $(date)"
