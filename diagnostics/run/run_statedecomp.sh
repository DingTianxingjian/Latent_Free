#!/bin/bash
#SBATCH --job-name=lf_sd
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH --time=01:30:00
#SBATCH --output=/lustre/fs1/home/ti874297/lf/latent_free/logs/%x_%j.out
#SBATCH --error=/lustre/fs1/home/ti874297/lf/latent_free/logs/%x_%j.out
#
# Item 1: state decomposition of the UniVLA LAM latent.  Reuses exp3's cached
# real-pair latents; encodes only the 3 shuffle tiers (same_ep/diff_ep/rand),
# both pre-VQ and post-VQ, then probes z -> action.  Latents cached per tier so
# a rerun is cheap.
set -euo pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free
PY=/lustre/fs1/home/ti874297/lf/envs/latent_free/bin/python
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export LF_RIDGE_ALPHAS="0.01,0.1,1,10,100,1000,10000"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  started $(date -Is)"
echo "git commit $(git rev-parse HEAD)  seed 0"
$PY exp3_statedecomp.py
echo "finished $(date -Is)"
