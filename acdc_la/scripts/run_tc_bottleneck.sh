#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 16 --mem=120G
#SBATCH -t 12:00:00
#SBATCH -J acdc_tcbn
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/tcbn_%j.out
# Low-bottleneck sweep (2026-09-25): tc_cont's recipe with ONLY the latent size changed.  LD = per-token latent dim;
# 4 tokens, so LD=16 -> 64-d and LD=4 -> 16-d (CoMo's size).  Same cosine schedule as tc_cont (--steps 128000) and
# stopped where tc_cont was (--stop-at 65000); --resume makes a walltime-cut leg restartable with the same command.
# Writes ckpt/tc_cont_d${LD}.pt and results/tc_cont_d${LD}_val.npz.
source /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/scripts/common.sh
LD=${LD:-64}; TC=${TC:-endpointdyn}
TAG=${TAG:-tc_cont_d${LD}}       # TC=endpoint TAG=tc_endpoint: the CoMo-style [F_0, N] arm (phase-2 arm C)
COMMON="--tc $TC --bottleneck continuous --n-tokens 4 --codebook 16 --latent-dim $LD --width 768 --enc-layers 6
        --heads 12 --layers 4 --d 64 --d-u 8 --beta-z 2e-4 --beta-u 0.1 --lam-commit 0.25 --restart-every 5000
        --pool acdc --modes 2 --lr 3e-4 --warmup 4000 --seed 0 --stride 4 --val-per-source 400"
if [ ! -f acdc_la/ckpt/${TAG}.pt ]; then          # smoke only on a fresh start
  run stage2_tc.train --tag smoke_${TAG} $COMMON --steps 40 --warmup 10 --limit-eps 200 --val-per-source 20 --batch 8 --workers 8 || exit 1
fi
run stage2_tc.train --tag $TAG $COMMON --steps 128000 --stop-at 65000 --batch 64 --save-every 5000 --workers 14 --resume
echo "TC BOTTLENECK $TAG DONE $(date)"
