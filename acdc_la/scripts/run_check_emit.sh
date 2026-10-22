#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 4 --mem=100G
#SBATCH -t 00:20:00
#SBATCH -J acdc_emit
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/emit_%j.out
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free
export PYTHONPATH=/lustre/fs1/home/ti874297/lf/latent_free:/lustre/fs1/home/ti874297/lf/latent_free/third_party/UniVLA TORCH_HOME=/lustre/fs1/home/ti874297/lf/latent_free/cache/torch PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
for R in ${RUNS:?set RUNS}; do
  /lustre/fs1/home/ti874297/lf/envs/univla_vla/bin/python -u -m acdc_la.downstream.vla.check_emit --ckpt /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/ckpt/vla_runs/$R 2>&1 | grep -E "^\[emit\]|Error|Traceback"
done
echo "EMIT DONE $(date)"
