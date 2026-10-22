#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH -N 1 -n 1 -c 16 --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J oxe_full
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/logs/oxe_full_%j.out
# UNCAPPED phase-1 caches on Newton (2026-10-16): every episode of each source in NAMES -> cache/oxe_full/<name>
# (a symlink into /lustre/fs1/groups/ytian), read by acdc_la.data.oxe_mix with ACDC_FULL_DATA=1.  CPU only.
# Skips sources already converted (meta.json present).
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free/phase1/oxe
export OXE_OUT=/lustre/fs1/home/ti874297/lf/latent_free/cache/oxe_full CUDA_VISIBLE_DEVICES=
for NAME in $NAMES; do
  [ -f $OXE_OUT/$NAME/meta.json ] && { echo "[skip] $NAME already converted"; continue; }
  /lustre/fs1/home/ti874297/lf/envs/univla_vla/bin/python -u oxe_convert.py --name "$NAME" --max-eps 100000000 --threads 16 2>&1 \
    | grep --line-buffered -vE "WARNING: All log|I0000|absl|cpu_feature_guard|rebuild TensorFlow|AVX2|cuda_diagnostics|tensorflow/"; rc=${PIPESTATUS[0]}
  [ $rc -ne 0 ] && echo "[FAIL] $NAME exit $rc"
done
echo "OXE FULL CONVERT JOB DONE $(date)"
