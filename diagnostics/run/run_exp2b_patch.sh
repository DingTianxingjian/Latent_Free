#!/bin/bash
#SBATCH --job-name=lf_2bp
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out
#SBATCH --error=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out
#
# Exp 2B on patch tokens.  Pure CPU linear algebra on cached clean patch
# features, but the `cis250950-gpu`/`-ai` accounts are pinned to GPU partitions
# (no CPU allocation exists), so it runs on `ai` and simply leaves the GPU idle.
# Short (~5 min real) with a 30-min wall, so it backfills ahead of longer jobs.
# The real cost is the 12288-d pg4 ridge Gram + randomized PCA, a few minutes.
set -euo pipefail
cd /anvil/scratch/x-tding2/latent_free
PY=/anvil/scratch/x-tding2/envs/latent_free/bin/python

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
# widen the ridge grid: pg4's full-D optimum sits at ~1e4 (see exp4), while the
# top-k projected probes want smaller alphas -- one grid has to cover both.
export LF_RIDGE_ALPHAS="0.01,0.1,1,10,100,1000,10000"

echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  started $(date -Is)"
echo "git commit $(git rev-parse HEAD)  seed 0  alphas=$LF_RIDGE_ALPHAS"
$PY exp2b_patch.py
echo "finished $(date -Is)"
