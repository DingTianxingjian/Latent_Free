#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 4 --mem=100G
#SBATCH -t 00:20:00
#SBATCH -J acdc_emit
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/emit_%j.out
set -u -o pipefail
cd /anvil/scratch/x-tding2/latent_free
export PYTHONPATH=/anvil/scratch/x-tding2/latent_free:/anvil/scratch/x-tding2/latent_free/third_party/UniVLA TORCH_HOME=/anvil/scratch/x-tding2/latent_free/cache/torch PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
for R in ${RUNS:?set RUNS}; do
  /anvil/scratch/x-tding2/envs/univla_vla/bin/python -u -m acdc_la.downstream.vla.check_emit --ckpt /anvil/scratch/x-tding2/latent_free/acdc_la/ckpt/vla_runs/$R 2>&1 | grep -E "^\[emit\]|Error|Traceback"
done
echo "EMIT DONE $(date)"
