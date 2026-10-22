#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 8 --mem=60G
#SBATCH -t 01:15:00
#SBATCH -J acdc_effect
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/effect_%j.out
# Review 2026-09-12: (1) downstream reads LATP's OUTPUT effect (+E), oracle z vs predicted z; (2) does the decoder use u.
source /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/scripts/common.sh
run downstream.libero_offline --tags s1_acdc_1ep s3_latp_1ep s1_mean_1ep --no-univla --steps 2000 --out libero_effect
run stage2_disentangle.ushuffle --tags s2_tcti_bu s2_aug_bu s2_tcti s1_acdc
echo "EFFECT READOUT DONE $(date)"
