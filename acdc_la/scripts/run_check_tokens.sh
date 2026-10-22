#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 4 --mem=60G
#SBATCH -t 00:30:00
#SBATCH -J acdc_chktok
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/chktok_%j.out
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free
export PYTHONPATH=/lustre/fs1/home/ti874297/lf/latent_free TORCH_HOME=/lustre/fs1/home/ti874297/lf/latent_free/cache/torch
/lustre/fs1/home/ti874297/lf/envs/latent_free/bin/python -u -m acdc_la.downstream.vla.check_univla_tokens 2>&1 | grep --line-buffered -vE "xFormers|warnings.warn|UserWarning|Using cache|swiglu|attention.py|block.py"
echo "CHKTOK exit ${PIPESTATUS[0]} $(date)"
