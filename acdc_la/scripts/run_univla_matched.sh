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
run baselines.univla --arm s2mix_1ep --tag univla_matched --workers 8 || exit 1
run stage1_preserve.probes --code-only univla_matched univla_official s1_acdc_1ep s1_mean_1ep
run stage2_disentangle.probes --no-ref univla_matched univla_official s1_acdc_1ep
echo "UNIVLA MATCHED DONE $(date)"
