#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 16 --mem=96G
#SBATCH -t 05:00:00
#SBATCH -J oxe_conv
#SBATCH -o /anvil/scratch/x-tding2/latent_free/logs/oxe_conv_%j.out
# Converts every dataset in NAMES sequentially (only a GPU partition is allocated to us, so
# one job with several datasets beats one job per dataset).  Skips names whose fetch is not
# complete or whose cache already exists.
set -u -o pipefail
cd /anvil/scratch/x-tding2/latent_free/phase1/oxe
for NAME in $NAMES; do
  [ -f /anvil/scratch/x-tding2/latent_free/cache/oxe/$NAME/meta.json ] && { echo "[skip] $NAME already converted"; continue; }
  ls /anvil/scratch/x-tding2/latent_free/data/oxe_raw/$NAME/*/FETCH_DONE >/dev/null 2>&1 || { echo "[skip] $NAME fetch not complete"; continue; }
  /anvil/scratch/x-tding2/envs/bridge/bin/python -u oxe_convert.py --name "$NAME" --max-eps "${MAX_EPS:-5000}" --threads 16 2>&1 \
    | grep -vE "WARNING: All log|I0000|absl|cpu_feature_guard|rebuild TensorFlow|AVX2"; rc=${PIPESTATUS[0]}; [ $rc -ne 0 ] && echo "[FAIL] $NAME exit $rc"
done
echo "OXE CONVERT JOB DONE $(date)"
