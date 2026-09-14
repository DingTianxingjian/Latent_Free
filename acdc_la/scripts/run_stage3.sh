#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 16 --mem=120G
#SBATCH -t 03:00:00
#SBATCH -J acdc_s3
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/s3_%j.out
# Stage III: LATP from a frozen tokenizer.  INIT = Stage-I/II tag (default s1_acdc; legacy rec_src reproduces swap_fz).
# Arms: method (frozen encoder), decoder-only control (no swap terms), full-swap ablation (encoder trained).
source /anvil/scratch/x-tding2/latent_free/acdc_la/scripts/common.sh
STEPS=${STEPS:-10000}; SEED=${SEED:-0}; INIT=${INIT:-s1_acdc}
RUNS=${RUNS:-"s3_latp:;s3_ctrl:--lam-real 0 --lam-cyc 0;s3_full:--train-encoder"}
run stage3_transport.train --tag smoke_s3 --init-from $INIT --steps 40 --warmup 10 --limit-eps 200 --val-per-source 20 --workers 8 || exit 1
IFS=';' read -ra LIST <<< "$RUNS"
for R in "${LIST[@]}"; do IFS=: read TAG EXTRA <<< "$R"
  run stage3_transport.train --tag ${TAG}${SUFFIX:-} --init-from $INIT --steps $STEPS --seed $SEED --workers 14 $EXTRA
done
TAGS=$(for R in "${LIST[@]}"; do IFS=: read TAG EXTRA <<< "$R"; echo -n "${TAG}${SUFFIX:-} "; done)
run stage3_transport.probes $INIT $TAGS; run stage3_transport.recovery $INIT $TAGS
echo "STAGE3 DONE $(date)"
