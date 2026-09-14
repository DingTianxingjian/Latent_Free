#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 8 --mem=120G
#SBATCH -t 01:00:00
#SBATCH -J acdc_vlasmoke
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/vlasmoke_%j.out
# Smoke of the controlled UniVLA fine-tuning: both arms, 12 optimiser steps, batch 2, then a merged checkpoint each.
set -u -o pipefail
cd /anvil/scratch/x-tding2/latent_free
export PYTHONPATH=/anvil/scratch/x-tding2/latent_free:/anvil/scratch/x-tding2/latent_free/third_party/UniVLA WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false TORCH_HOME=/anvil/scratch/x-tding2/latent_free/cache/torch PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
PY=/anvil/scratch/x-tding2/envs/univla_vla/bin/python
for L in ${ARMS:-acdc univla}; do
  $PY -m torch.distributed.run --standalone --nproc-per-node 1 -m acdc_la.downstream.vla.finetune_libero_ctrl --latent $L --max_steps 12 --save_steps 12 --batch_size 2 --grad_accumulation_steps 1 --shuffle_buffer_size 200 --log_every 2 --run_id_note smoke 2>&1 | grep --line-buffered -vE "xFormers|warnings.warn|UserWarning|tensorflow/|I0000|E0000|W0000|oneDNN|TF-TRT|cuda_"
  echo "SMOKE $L exit ${PIPESTATUS[0]} $(date)"
done
echo "VLA SMOKE DONE $(date)"
