#!/bin/bash
#SBATCH --job-name=latent_free
#SBATCH --account=cis250950-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out
#SBATCH --error=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out

set -euo pipefail
cd /anvil/scratch/x-tding2/latent_free

PY=/anvil/scratch/x-tding2/envs/latent_free/bin/python
export TORCH_HOME=/anvil/scratch/x-tding2/latent_free/cache/torch
export HF_HOME=/anvil/scratch/x-tding2/hf_cache
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export PYTHONUNBUFFERED=1

echo "=================================================================="
echo "host        : $(hostname)"
echo "job         : ${SLURM_JOB_ID:-none}"
echo "started     : $(date -Is)"
echo "git commit  : $(git -C /anvil/scratch/x-tding2/latent_free rev-parse HEAD 2>/dev/null || echo 'not a repo')"
echo "python      : $PY"
echo "seed        : 0 (episode-level 80/20 split, per task)"
echo "=================================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo; echo "########## dataset stats ##########"
$PY data.py

echo; echo "########## stage 1: feature extraction (GPU) ##########"
$PY features.py --examples --conditions clean patch patch_static border border_static

echo; echo "########## stage 2: exp1 + exp2 (parallel, CPU) ##########"
$PY exp1.py > logs/exp1_${SLURM_JOB_ID:-local}.log 2>&1 &
E1=$!
$PY exp2.py > logs/exp2_${SLURM_JOB_ID:-local}.log 2>&1 &
E2=$!
wait $E1; R1=$?
wait $E2; R2=$?

echo "--- exp1 log ---"; cat logs/exp1_${SLURM_JOB_ID:-local}.log
echo "--- exp2 log ---"; cat logs/exp2_${SLURM_JOB_ID:-local}.log
echo "exp1 exit=$R1  exp2 exit=$R2"

echo; echo "########## outputs ##########"
ls -la results/ figs/
echo "finished    : $(date -Is)"
exit $(( R1 + R2 ))
