#!/bin/bash
# Local (UCF workstation) equivalent of acdc_la/scripts/run_phys_replay.sh
cd /home/ti874297/Projects/latent_free
export PYTHONPATH=/home/ti874297/Projects/latent_free:/home/ti874297/Projects/latent_free/third_party/UniVLA:/home/ti874297/Projects/latent_free/envs/LIBERO
export WANDB_MODE=disabled PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LF_ROOT=/home/ti874297/Projects/latent_free
export LF_LIBERO_HDF5=/data/ti874297/LIBERO_hdf5
export LF_PHYS_OUT=/home/ti874297/Projects/latent_free/acdc_la/results/phys_replay
export TORCH_HOME=/data/ti874297/cache/torch
mkdir -p "$LF_PHYS_OUT"
exec envs/univla_vla/bin/python -u -m acdc_la.downstream.phys_replay "$@" 2>&1 | grep --line-buffered -vE "Warning|warn\(|robosuite_macros"
