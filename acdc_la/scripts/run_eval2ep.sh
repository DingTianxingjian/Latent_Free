#!/bin/bash
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 8 --mem=60G
#SBATCH -t 01:00:00
#SBATCH -J acdc_eval2ep
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/eval2ep_%j.out
# Quick pretraining result: the 2-epoch checkpoints (265k / 260k steps) of both chains, full 400-clip/source dump + Stage I probes.
source /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/scripts/common.sh
run stage1_preserve.train --tag s1_acdc_2ep --eval-only --workers 8
run stage1_preserve.train --tag s1_mean_2ep --eval-only --pool mean --workers 8
run stage1_preserve.probes --code-only s1_acdc_2ep s1_mean_2ep s1_acdc_1ep s1_mean_1ep univla_matched
run stage3_transport.probes s1_acdc_2ep s1_mean_2ep
echo "EVAL2EP DONE $(date)"
