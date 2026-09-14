#!/bin/bash
#SBATCH --job-name=lf_prlax
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=00:50:00
#SBATCH --output=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out
#SBATCH --error=/anvil/scratch/x-tding2/latent_free/logs/%x_%j.out
#
# PRLA feature extraction: V-JEPA2 fused latents for history/future windows as
# SEPARATE clips (causality).  Reuses the Exp-6b fused path (verified rel=0.000).
set -euo pipefail
cd /anvil/scratch/x-tding2/latent_free/phase1
PY=/anvil/scratch/x-tding2/envs/latent_free/bin/python
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  started $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
$PY extract_prla.py ${PRLAX_ARGS:-}
echo "finished $(date -Is)"
