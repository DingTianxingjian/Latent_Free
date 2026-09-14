#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 12 --mem=90G
#SBATCH -t 30:00:00
#SBATCH -J acdc_long
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/long_%j.out
# FORMAL pretraining budget: one arm per job, ~1 epoch of the 8.19M-clip mixture.
#   STAGE=1|2|3  TAG=<tag>  STEPS (default 128000)  BATCH (default 64)  INIT (stage 2/3)  EXTRA (flags)
# 128k x 64 = 8.19M clips = 1.0 epoch (the 10k x 32 pilot runs were 0.039 epoch).  Checkpoints every 5000 steps;
# re-submit the same command with RESUME=1 after a walltime kill and it continues from the last checkpoint.
source /anvil/scratch/x-tding2/latent_free/acdc_la/scripts/common.sh
STEPS=${STEPS:-128000}; BATCH=${BATCH:-64}; SEED=${SEED:-0}; RES=""; [ "${RESUME:-0}" = 1 ] && RES="--resume"
case "$STAGE" in
  1) MOD=stage1_preserve.train; INITF="";;
  2) MOD=stage2_disentangle.train; INITF="--init-from $INIT";;
  3) MOD=stage3_transport.train; INITF="--init-from $INIT";;
  *) echo "STAGE must be 1|2|3"; exit 2;;
esac
run $MOD --tag $TAG --steps $STEPS --batch $BATCH --seed $SEED --workers 14 --warmup 4000 $INITF ${EXTRA:-} $RES
echo "LONG $TAG exit $? $(date)"
