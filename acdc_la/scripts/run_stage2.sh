#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 16 --mem=120G
#SBATCH -t 04:00:00
#SBATCH -J acdc_s2
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/s2_%j.out
# Stage II: TC/TI by intervention, from a Stage-I tokenizer.  INIT = Stage-I tag (default s1_acdc; legacy rec_src works).
# Arms: the method, the augmentation control (same data, no separation terms), and the two single-term ablations.
source /anvil/scratch/x-tding2/latent_free/acdc_la/scripts/common.sh
STEPS=${STEPS:-10000}; SEED=${SEED:-0}; INIT=${INIT-s1_acdc}   # INIT="" (set but empty) = train from scratch
RUNS=${RUNS:-"s2_tcti:;s2_aug:--lam-inv 0 --lam-xrec 0;s2_inv:--lam-xrec 0;s2_xrec:--lam-inv 0"}
INITF=""; [ -n "$INIT" ] && INITF="--init-from $INIT"
run stage2_disentangle.train --tag smoke_s2 $INITF --steps 40 --warmup 10 --limit-eps 200 --val-per-source 20 --workers 8 || exit 1
IFS=';' read -ra LIST <<< "$RUNS"
for R in "${LIST[@]}"; do IFS=: read TAG EXTRA <<< "$R"
  run stage2_disentangle.train --tag ${TAG}${SUFFIX:-} $INITF --steps $STEPS --seed $SEED --workers 14 $EXTRA
done
run stage2_disentangle.probes $(for R in "${LIST[@]}"; do IFS=: read TAG EXTRA <<< "$R"; echo -n "${TAG}${SUFFIX:-} "; done)
echo "STAGE2 DONE $(date)"
