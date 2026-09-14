#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 16 --mem=120G
#SBATCH -t 01:30:00
#SBATCH -J acdc_univla
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/univla_%j.out
# UniVLA stage-2 codes (released weights) on the same validation clips -> results/univla_official_val.npz
source /anvil/scratch/x-tding2/latent_free/acdc_la/scripts/common.sh
run baselines.univla --arm ${ARM:-official_stage2} --tag ${TAG:-univla_official} --workers 8
run stage1_preserve.probes ${TAG:-univla_official}; run stage2_disentangle.probes ${TAG:-univla_official}
echo "UNIVLA DUMP DONE $(date)"
