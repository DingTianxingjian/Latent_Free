#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 4 --mem=90G
#SBATCH -t 01:00:00
#SBATCH -J acdc_gatelite
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/gatelite_%j.out
# The cheap half of run_tc_gate.sh: write the validation dump from whatever checkpoint exists, then the OXE probes
# and the within-family predictability.  Drops the LIBERO oracle pass, which is the slow part.  One hour and four
# cores backfills; the three-hour eight-core gate was estimated four days out.
source /anvil/scratch/x-tding2/latent_free/acdc_la/scripts/common.sh
for T in ${TAGS:?set TAGS}; do
  [ -f acdc_la/results/${T}_val.npz ] || run stage2_tc.train --tag $T --eval-only --workers 4 || exit 1
done
run stage1_preserve.probes --code-only $TAGS s1_acdc_1ep s1_mean_1ep
run common.predictability $TAGS s1_acdc_1ep s1_mean_1ep
echo "GATE LITE DONE $(date)"
