#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH -N 1 -n 1 -c 2 --mem=8G
#SBATCH -t 3-00:00:00
#SBATCH -J p2fft_midcheck
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/p2fft_midcheck_%j.out
# Intermediate check of the full-parameter phase-2 run (2026-10-27): when its sharded checkpoint reaches STEP, take a
# hard-link snapshot (the trainer's next save cannot delete it), then queue (1) consolidation to a vla_full.pt run dir and
# (2) 10 % LIBERO-Long post-training + 500-episode evaluation from it -- the SAME downstream recipe as the 1/10-scale
# phase-2 cell (61.6 %), so the only difference is phase 2.  CPU-only watcher.
set -u
cd /lustre/fs1/home/ti874297/lf/latent_free
STEP=${STEP:-5000}; SUITE=${SUITE:-libero_10}
R="acdc_la/ckpt/vla_runs/tc_cont+acdc_oxe29+b256x4+lr2e-05+fft+s20000+fm+q4--p2fft_ours"
PY=/lustre/fs1/home/ti874297/lf/envs/univla_vla/bin/python
SNAP=acdc_la/ckpt/vla_runs/_snapshots/p2fft_ours_step$STEP; OUT=acdc_la/ckpt/vla_runs/p2fft_ours_step$STEP
while true; do
  S=$($PY -c "import torch; print(torch.load('$R/trainer_state.pt', map_location='cpu')['step'])" 2>/dev/null)
  if [ "${S:-0}" -ge "$STEP" ]; then break; fi
  sleep 300
done
[ "$S" = "$STEP" ] || echo "[midcheck] WARNING: checkpoint is at step $S, not $STEP (snapshotting $S)"
mkdir -p $(dirname $SNAP); rm -rf $SNAP; cp -al "$R/fsdp_ckpt" $SNAP && echo "[midcheck] snapshot of step $S at $SNAP $(date)"
JC=$(sbatch --parsable -A ytian -p normal -N1 -n1 -c 8 --mem=120G -t 02:00:00 -J p2fft_consolidate -o acdc_la/logs/p2fft_consolidate_%j.out \
     --wrap "cd /lustre/fs1/home/ti874297/lf/latent_free && export PYTHONPATH=\$PWD:\$PWD/third_party/UniVLA CUDA_VISIBLE_DEVICES= && $PY acdc_la/scripts/consolidate_fft_ckpt.py --ckpt $SNAP --run '$R' --out $OUT --step $S")
JP=$(LATENT=tc_cont INIT=$OUT SUITE=$SUITE NOTE=p2fft${S}_${SUITE}_frac10 sbatch --parsable --dependency=afterok:$JC -J p3_p2fft${S} \
     acdc_la/scripts/run_vla_post2_newton.sh)
echo "[midcheck] consolidation $JC -> post-training + eval $JP ($SUITE, 10 %) $(date)"
