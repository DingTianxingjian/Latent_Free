#!/bin/bash
#SBATCH --job-name=lf_p1
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH --time=00:45:00
#SBATCH --output=/lustre/fs1/home/ti874297/lf/latent_free/logs/%x_%j.out
#SBATCH --error=/lustre/fs1/home/ti874297/lf/latent_free/logs/%x_%j.out
#
# Phase 1 -- LAM baseline (and later PRLA) on frozen V-JEPA2 g4 features.
# Pass args via PHASE1_ARGS, e.g. sbatch --export=ALL,PHASE1_ARGS="--smoke" run_phase1.sh
set -euo pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free/phase1
PY=/lustre/fs1/home/ti874297/lf/envs/latent_free/bin/python
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export LF_RIDGE_ALPHAS="0.01,0.1,1,10,100,1000,10000"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  started $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "git commit $(git -C /lustre/fs1/home/ti874297/lf/latent_free rev-parse HEAD 2>/dev/null || echo none)"
$PY train.py ${PHASE1_ARGS:-}
echo "finished $(date -Is)"
