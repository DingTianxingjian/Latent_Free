#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 4 --mem=60G
#SBATCH -t 00:30:00
#SBATCH -J acdc_chktok
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/chktok_%j.out
set -u -o pipefail
cd /anvil/scratch/x-tding2/latent_free
export PYTHONPATH=/anvil/scratch/x-tding2/latent_free TORCH_HOME=/anvil/scratch/x-tding2/latent_free/cache/torch
/anvil/scratch/x-tding2/envs/latent_free/bin/python -u -m acdc_la.downstream.vla.check_univla_tokens 2>&1 | grep --line-buffered -vE "xFormers|warnings.warn|UserWarning|Using cache|swiglu|attention.py|block.py"
echo "CHKTOK exit ${PIPESTATUS[0]} $(date)"
