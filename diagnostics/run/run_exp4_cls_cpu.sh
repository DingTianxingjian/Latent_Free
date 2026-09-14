#!/bin/bash
# Early preview: the exp-4 matrix restricted to the CLS space, which needs NO new
# features -- cache/feats_clean.npy already exists.  Runs GPU-free, so it can use
# the idle CPUs on the ai nodes while every GPU on the cluster is allocated.
#
# Writes to a _cpupreview suffix on purpose: the authoritative table comes from
# the single GPU run, so these rows are never merged into it.
#SBATCH --job-name=lf_exp4_cls
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out
#SBATCH --error=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out

set -euo pipefail
cd /anvil/scratch/x-tding2/latent_free

PY=/anvil/scratch/x-tding2/envs/latent_free/bin/python
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1
export LF_OUT_SUFFIX=_cpupreview

echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  started $(date -Is)"
echo "git commit $(git rev-parse HEAD)  seed 0  CPU-only preview, cls space"

$PY exp4_patch.py --condition clean --spaces cls --device cpu --budget-min 50
echo "finished $(date -Is)"
