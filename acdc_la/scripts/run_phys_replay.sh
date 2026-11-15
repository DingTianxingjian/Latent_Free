#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=preemptable
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --exclude=evc11
#SBATCH -N 1 -n 1 -c 8 --mem=48G
#SBATCH -t 4:00:00
#SBATCH -J phys_replay
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/phys_replay_%j.out
# MODE=probe|collect ARGS="..." sbatch acdc_la/scripts/run_phys_replay.sh
export PYTHONPATH=/lustre/fs1/home/ti874297/lf/latent_free:/lustre/fs1/home/ti874297/lf/latent_free/third_party/UniVLA:/lustre/fs1/home/ti874297/lf/envs/LIBERO WANDB_MODE=disabled TORCH_HOME=/lustre/fs1/home/ti874297/lf/latent_free/cache/torch PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd /lustre/fs1/home/ti874297/lf/latent_free
/lustre/fs1/home/ti874297/lf/envs/univla_vla/bin/python -u -m acdc_la.downstream.phys_replay ${MODE:-probe} $ARGS 2>&1 | grep --line-buffered -vE "Warning|warn\(|robosuite_macros"
