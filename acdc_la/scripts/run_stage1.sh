#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 16 --mem=120G
#SBATCH -t 04:00:00
#SBATCH -J acdc_s1
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/s1_%j.out
# Stage I tokenizers, sequential in one job.  RUNS = ';'-separated "tag:extra-args".
# Default: the AC/DC tokenizer and its pooled ablations on the same data, seed and steps; every run ends with a
# validation dump results/<tag>_val.npz, read by `python -m acdc_la.stage1_preserve.probes <tags>` (CPU).
source /anvil/scratch/x-tding2/latent_free/acdc_la/scripts/common.sh
STEPS=${STEPS:-10000}; SEED=${SEED:-0}
RUNS=${RUNS:-"s1_acdc:--modes 2;s1_dc:--modes 1;s1_mean:--pool mean;s1_acdc3:--modes 3"}
run stage1_preserve.train --tag smoke_s1 --steps 40 --warmup 10 --limit-eps 200 --val-per-source 20 --workers 8 || exit 1
IFS=';' read -ra LIST <<< "$RUNS"
for R in "${LIST[@]}"; do IFS=: read TAG EXTRA <<< "$R"
  run stage1_preserve.train --tag ${TAG}${SUFFIX:-} --steps $STEPS --seed $SEED --workers 14 $EXTRA
done
run stage1_preserve.probes $(for R in "${LIST[@]}"; do IFS=: read TAG EXTRA <<< "$R"; echo -n "${TAG}${SUFFIX:-} "; done)
echo "STAGE1 DONE $(date)"
