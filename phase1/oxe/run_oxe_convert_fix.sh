#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 16 --mem=96G
#SBATCH -t 04:00:00
#SBATCH -J oxe_conv2
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/logs/oxe_conv2_%j.out
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free/phase1/oxe
PY=/lustre/fs1/home/ti874297/lf/envs/bridge/bin/python
F="grep -vE WARNING:.All.log|I0000|absl|cpu_feature_guard|rebuild.TensorFlow|AVX2"
OUT=/lustre/fs1/home/ti874297/lf/latent_free/cache/oxe
if [ "${SKIP_REGRESSION:-0}" != "1" ]; then
# 1. regression: reconvert ucsd_kitchen with the SkipDecoding path into a temp name and compare frames byte-for-byte
OUT=/lustre/fs1/home/ti874297/lf/latent_free/cache/oxe
mv $OUT/ucsd_kitchen_dataset_converted_externally_to_rlds $OUT/_ucsd_old
$PY -u oxe_convert.py --name ucsd_kitchen_dataset_converted_externally_to_rlds --threads 16 2>&1 | grep -vE "WARNING: All log|I0000|absl|cpu_feature_guard|rebuild TensorFlow|AVX2"
$PY - <<'PYEOF'
import numpy as np, sys
O="/lustre/fs1/home/ti874297/lf/latent_free/cache/oxe/"
a=np.load(O+"_ucsd_old/frames.npy",mmap_mode="r"); b=np.load(O+"ucsd_kitchen_dataset_converted_externally_to_rlds/frames.npy",mmap_mode="r")
za=np.load(O+"_ucsd_old/index.npz"); zb=np.load(O+"ucsd_kitchen_dataset_converted_externally_to_rlds/index.npz")
same_frames = a.shape==b.shape and all(np.array_equal(a[i:i+500],b[i:i+500]) for i in range(0,len(a),500))
same_idx = all(np.array_equal(za[k],zb[k]) for k in ("starts","lengths","is_val","actions"))
print(f"[regression] frames identical={same_frames} index identical={same_idx}")
sys.exit(0 if (same_frames and same_idx) else 2)
PYEOF
rc=$?; if [ $rc -ne 0 ]; then echo "REGRESSION FAILED: SkipDecoding path differs from the tfds-decoded cache -- stopping"; rm -rf $OUT/ucsd_kitchen_dataset_converted_externally_to_rlds; mv $OUT/_ucsd_old $OUT/ucsd_kitchen_dataset_converted_externally_to_rlds; exit 2; fi
rm -rf $OUT/_ucsd_old; echo "regression OK"
fi
# 2. the four sources that failed or were never attempted
for NAME in ${NAMES:-iamlab_cmu_pickup_insert_converted_externally_to_rlds furniture_bench_dataset_converted_externally_to_rlds}; do
  rm -rf $OUT/$NAME
  $PY -u oxe_convert.py --name "$NAME" --threads 16 2>&1 | grep -vE "WARNING: All log|I0000|absl|cpu_feature_guard|rebuild TensorFlow|AVX2"; rc=${PIPESTATUS[0]}; [ $rc -ne 0 ] && echo "[FAIL] $NAME exit $rc"
done
echo "OXE CONVERT FIX JOB DONE $(date)"
