#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 16 --mem=200G
#SBATCH -t 01:30:00
#SBATCH -J acdc_p2smoke
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/p2smoke_%j.out
# Phase-2 smoke on the OXE mixture (never run against data/oxe_raw before): 20 steps per arm on ONE GPU, micro-batch 8,
# with GPU memory sampled every minute.  Checks the labeller on OXE frames, the fixed-window patch, the acdc_oxe29
# mixture, and the memory a single card needs.  Both arms must exit 0 before the 30k-step runs.
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free
for ARM in "tc_cont p2smoke_ours" "univla_cont p2smoke_univla"; do
  set -- $ARM
  LATENT=$1 NOTE=$2 NGPU=1 STEPS=20 SAVE=1000 EXTRA="--grad_accumulation_steps 1" \
    bash acdc_la/scripts/run_vla_phase2.sh || { echo "SMOKE FAILED on $1"; exit 1; }
done
echo "PHASE2 SMOKE OK $(date)"
