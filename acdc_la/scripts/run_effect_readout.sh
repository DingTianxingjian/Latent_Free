#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 8 --mem=60G
#SBATCH -t 01:15:00
#SBATCH -J acdc_effect
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/effect_%j.out
# Review 2026-09-12: (1) downstream reads LATP's OUTPUT effect (+E), oracle z vs predicted z; (2) does the decoder use u.
source /anvil/scratch/x-tding2/latent_free/acdc_la/scripts/common.sh
run downstream.libero_offline --tags s1_acdc_1ep s3_latp_1ep s1_mean_1ep --no-univla --steps 2000 --out libero_effect
run stage2_disentangle.ushuffle --tags s2_tcti_bu s2_aug_bu s2_tcti s1_acdc
echo "EFFECT READOUT DONE $(date)"
