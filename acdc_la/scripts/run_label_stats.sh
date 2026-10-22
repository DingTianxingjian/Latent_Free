#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 8 --mem=100G
#SBATCH -t 01:00:00
#SBATCH -J acdc_labstat
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/labstat_%j.out
# What the VLA token accuracies mean: majority floor, entropy and augmentation stability of each arm's labels.
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free
export PYTHONPATH=/lustre/fs1/home/ti874297/lf/latent_free:/lustre/fs1/home/ti874297/lf/latent_free/third_party/UniVLA WANDB_MODE=disabled TORCH_HOME=/lustre/fs1/home/ti874297/lf/latent_free/cache/torch PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
/lustre/fs1/home/ti874297/lf/envs/univla_vla/bin/python -u -m acdc_la.downstream.vla.label_stats --arms ${ARMS:-univla acdc} --n ${N:-1000} 2>&1 | grep --line-buffered -vE "xFormers|warnings.warn|UserWarning|tensorflow/|I0000|E0000|W0000|oneDNN|TF-TRT|cuda_|AtomicFunction|NoneType|Exception ignored|^Traceback|atomic_function"
echo "LABSTAT exit ${PIPESTATUS[0]} $(date)"
