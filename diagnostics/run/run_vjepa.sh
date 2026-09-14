#!/bin/bash
#SBATCH --job-name=lf_vj
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
# Item 3 (V-JEPA2): pred 4 -- is spatial pooling less harmful for a prediction
# encoder than for DINOv2.  VALIDATE=<n> runs only n episodes (contract check).
set -euo pipefail
cd /anvil/scratch/x-tding2/latent_free
PY=/anvil/scratch/x-tding2/envs/latent_free/bin/python
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export LF_RIDGE_ALPHAS="0.01,0.1,1,10,100,1000,10000"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  started $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "git commit $(git rev-parse HEAD)"

if [[ -n "${VALIDATE:-}" ]]; then
  echo "=== VALIDATION: $VALIDATE episodes (clean) ==="
  $PY features_vjepa.py --limit-episodes "$VALIDATE"
else
  echo "=== 1/2 extract V-JEPA2 features (clean + 4 distractor conditions) ==="
  $PY features_vjepa.py --conditions patch patch_static border border_static --redo
  echo "=== 2/2 pooled-vs-spatial decodability + null ==="
  $PY exp_vjepa.py
fi
echo "finished $(date -Is)"
