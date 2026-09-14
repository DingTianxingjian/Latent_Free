#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 8 --mem=100G
#SBATCH -t 03:00:00
#SBATCH -J acdc_libero
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/libero_%j.out
# Downstream offline accuracy on LIBERO-spatial: latent_acc (UniVLA's metric), action token accuracy (oracle / chain / no-latent).
source /anvil/scratch/x-tding2/latent_free/acdc_la/scripts/common.sh
run downstream.libero_offline --tags ${TAGS:-s1_acdc s1_mean s1_dc rec_src s3_latp} --out ${OUT:-libero_offline}
echo "LIBERO OFFLINE DONE $(date)"
