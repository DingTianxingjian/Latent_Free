#!/bin/bash
#SBATCH --job-name=lf_null
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=00:40:00
#SBATCH --output=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out
#SBATCH --error=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out
#
# Exp 2B null control: top-k vs random-k vs bottom-k action recovery on clean
# Delta-f.  Decides whether the patch-2B "action leads variance" is real or
# whether PCA ordering is uninformative.  SVD + projections on the GPU (already
# reserved), refit with the same sklearn probe as the 2B table.
set -euo pipefail
cd /anvil/scratch/x-tding2/latent_free
PY=/anvil/scratch/x-tding2/envs/latent_free/bin/python

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export LF_RIDGE_ALPHAS="0.01,0.1,1,10,100,1000,10000"

echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  started $(date -Is)"
echo "git commit $(git rev-parse HEAD)  seed 0  alphas=$LF_RIDGE_ALPHAS"
$PY exp2b_null.py
echo "finished $(date -Is)"
