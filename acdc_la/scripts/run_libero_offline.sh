#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 8 --mem=100G
#SBATCH -t 03:00:00
#SBATCH -J acdc_libero
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/libero_%j.out
# Downstream offline accuracy on LIBERO-spatial: latent_acc (UniVLA's metric), action token accuracy (oracle / chain / no-latent).
source /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/scripts/common.sh
run downstream.libero_offline --tags ${TAGS:-s1_acdc s1_mean s1_dc rec_src s3_latp} --out ${OUT:-libero_offline}
echo "LIBERO OFFLINE DONE $(date)"
