#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 8 --mem=60G
#SBATCH -t 01:30:00
#SBATCH -J univla_matched
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/univla_matched_%j.out
# Budget-matched UniVLA stage-2 (s2mix_1ep: their recipe, our data, 128k x 64): dump on the val clips + Stage I/II probes +
# LIBERO offline accuracy, so it sits in every table next to s1_acdc_1ep / s1_mean_1ep.
source /anvil/scratch/x-tding2/latent_free/acdc_la/scripts/common.sh
UARM=${UARM:-s2mix_1ep}; TAG=${TAG:-univla_matched}; REF=${REF:-"s1_acdc_1ep s1_mean_1ep"}
run baselines.univla --arm $UARM --tag $TAG --workers 8 || exit 1
run stage1_preserve.probes --code-only $TAG univla_official $REF
run stage2_disentangle.probes --no-ref $TAG univla_official $REF
echo "UNIVLA MATCHED DONE $(date)"
