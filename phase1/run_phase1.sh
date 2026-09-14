#!/bin/bash
#SBATCH --job-name=lf_p1
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=00:45:00
#SBATCH --output=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out
#SBATCH --error=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out
#
# Phase 1 -- LAM baseline (and later PRLA) on frozen V-JEPA2 g4 features.
# Pass args via PHASE1_ARGS, e.g. sbatch --export=ALL,PHASE1_ARGS="--smoke" run_phase1.sh
set -euo pipefail
cd /anvil/scratch/x-tding2/latent_free/phase1
PY=/anvil/scratch/x-tding2/envs/latent_free/bin/python
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export LF_RIDGE_ALPHAS="0.01,0.1,1,10,100,1000,10000"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  started $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "git commit $(git -C /anvil/scratch/x-tding2/latent_free rev-parse HEAD 2>/dev/null || echo none)"
$PY train.py ${PHASE1_ARGS:-}
echo "finished $(date -Is)"
