#!/bin/bash
# exp1 + exp2 are pure CPU (sklearn on cached features), so they do not need to
# wait behind the GPU queue.  `debug` allows 2 h and is nearly empty.
#SBATCH --job-name=lf_analysis
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fs1/home/ti874297/lf/latent_free/logs/%x_%j.out
#SBATCH --error=/lustre/fs1/home/ti874297/lf/latent_free/logs/%x_%j.out

set -euo pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free

PY=/lustre/fs1/home/ti874297/lf/envs/latent_free/bin/python
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export PYTHONUNBUFFERED=1

echo "host $(hostname)  job ${SLURM_JOB_ID:-none}  started $(date -Is)"
echo "git commit $(git rev-parse HEAD)  seed 0"

for c in clean patch patch_static border border_static; do
  [ -f "cache/feats_${c}.npy" ] || { echo "MISSING cache/feats_${c}.npy -- run run_features_debug.sh first"; exit 1; }
done

$PY exp1.py > logs/exp1_${SLURM_JOB_ID:-local}.log 2>&1 &
E1=$!
$PY exp2.py > logs/exp2_${SLURM_JOB_ID:-local}.log 2>&1 &
E2=$!
set +e; wait $E1; R1=$?; wait $E2; R2=$?; set -e

echo "--- exp1 log ---"; cat logs/exp1_${SLURM_JOB_ID:-local}.log
echo "--- exp2 log ---"; cat logs/exp2_${SLURM_JOB_ID:-local}.log
echo "exp1 exit=$R1  exp2 exit=$R2"
ls -la results/ figs/*.png 2>/dev/null | head -30
echo "finished $(date -Is)"
exit $(( R1 + R2 ))
