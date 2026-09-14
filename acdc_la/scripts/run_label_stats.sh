#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 8 --mem=100G
#SBATCH -t 01:00:00
#SBATCH -J acdc_labstat
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/labstat_%j.out
# What the VLA token accuracies mean: majority floor, entropy and augmentation stability of each arm's labels.
set -u -o pipefail
cd /anvil/scratch/x-tding2/latent_free
export PYTHONPATH=/anvil/scratch/x-tding2/latent_free:/anvil/scratch/x-tding2/latent_free/third_party/UniVLA WANDB_MODE=disabled TORCH_HOME=/anvil/scratch/x-tding2/latent_free/cache/torch PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
/anvil/scratch/x-tding2/envs/univla_vla/bin/python -u -m acdc_la.downstream.vla.label_stats --arms ${ARMS:-univla acdc} --n ${N:-1000} 2>&1 | grep --line-buffered -vE "xFormers|warnings.warn|UserWarning|tensorflow/|I0000|E0000|W0000|oneDNN|TF-TRT|cuda_|AtomicFunction|NoneType|Exception ignored|^Traceback|atomic_function"
echo "LABSTAT exit ${PIPESTATUS[0]} $(date)"
