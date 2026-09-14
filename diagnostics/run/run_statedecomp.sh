#!/bin/bash
#SBATCH --job-name=lf_sd
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out
#SBATCH --error=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out
#
# Item 1: state decomposition of the UniVLA LAM latent.  Reuses exp3's cached
# real-pair latents; encodes only the 3 shuffle tiers (same_ep/diff_ep/rand),
# both pre-VQ and post-VQ, then probes z -> action.  Latents cached per tier so
# a rerun is cheap.
set -euo pipefail
cd /anvil/scratch/x-tding2/latent_free
PY=/anvil/scratch/x-tding2/envs/latent_free/bin/python
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export LF_RIDGE_ALPHAS="0.01,0.1,1,10,100,1000,10000"
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  started $(date -Is)"
echo "git commit $(git rev-parse HEAD)  seed 0"
$PY exp3_statedecomp.py
echo "finished $(date -Is)"
