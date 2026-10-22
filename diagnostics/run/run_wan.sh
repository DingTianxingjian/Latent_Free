#!/bin/bash
#SBATCH --job-name=lf_wan
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fs1/home/ti874297/lf/latent_free/logs/%x_%j.out
#SBATCH --error=/lustre/fs1/home/ti874297/lf/latent_free/logs/%x_%j.out
#
# Item 3 (Wan): extract Wan2.2-VAE latents for every LIBERO frame, then run the
# same decodability + null protocol as DINOv2.  Built-in smoke (4 frames) fails
# fast if the encode contract is wrong.  VALIDATE=<n> runs only n episodes.
set -euo pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free
PY=/lustre/fs1/home/ti874297/lf/envs/latent_free/bin/python
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export LF_RIDGE_ALPHAS="0.01,0.1,1,10,100,1000,10000"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  started $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "git commit $(git rev-parse HEAD)"

if [[ -n "${VALIDATE:-}" ]]; then
  echo "=== VALIDATION: $VALIDATE episodes ==="
  $PY features_wan.py --limit-episodes "$VALIDATE"
else
  echo "=== 1/2 extract Wan VAE features (clean cached; +4 distractor conditions) ==="
  $PY features_wan.py --conditions clean patch patch_static border border_static
  echo "=== 2/2 decodability + per-condition null (all 5 conditions) ==="
  $PY exp_wan.py
fi
echo "finished $(date -Is)"
