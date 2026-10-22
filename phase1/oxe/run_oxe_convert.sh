#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 16 --mem=96G
#SBATCH -t 05:00:00
#SBATCH -J oxe_conv
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/logs/oxe_conv_%j.out
# Converts every dataset in NAMES sequentially (only a GPU partition is allocated to us, so
# one job with several datasets beats one job per dataset).  Skips names whose fetch is not
# complete or whose cache already exists.
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free/phase1/oxe
for NAME in $NAMES; do
  [ -f /lustre/fs1/home/ti874297/lf/latent_free/cache/oxe/$NAME/meta.json ] && { echo "[skip] $NAME already converted"; continue; }
  ls /lustre/fs1/home/ti874297/lf/latent_free/data/oxe_raw/$NAME/*/FETCH_DONE >/dev/null 2>&1 || { echo "[skip] $NAME fetch not complete"; continue; }
  /lustre/fs1/home/ti874297/lf/envs/bridge/bin/python -u oxe_convert.py --name "$NAME" --max-eps "${MAX_EPS:-5000}" --threads 16 2>&1 \
    | grep -vE "WARNING: All log|I0000|absl|cpu_feature_guard|rebuild TensorFlow|AVX2"; rc=${PIPESTATUS[0]}; [ $rc -ne 0 ] && echo "[FAIL] $NAME exit $rc"
done
echo "OXE CONVERT JOB DONE $(date)"
