#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 16 --mem=120G
#SBATCH -t 03:00:00
#SBATCH -J acdc_pair
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/pair_%j.out
# Stage-I ablation: PAIR-WISE tokenizers (UniVLA's usage), same architecture / data / budget as s1_acdc.
#   s1_pair    T=2, d=32  -> 4 codes x 128 = 512 dims when concatenated over the 4 segments
#   s1_pair16  T=2, d=16  -> 256 dims (dimension-matched to the DC+AC1 clip code)
# Then each is applied per segment to the T=5 val clips and probed with the same Stage-I probes.
source /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/scripts/common.sh
STEPS=${STEPS:-10000}; SEED=${SEED:-0}
run stage1_preserve.train --tag s1_pair --T 2 --modes 1 --steps $STEPS --seed $SEED --workers 14 || exit 1
run stage1_preserve.train --tag s1_pair16 --T 2 --modes 1 --d 16 --steps $STEPS --seed $SEED --workers 14 || exit 1
run stage1_preserve.pair_dump --tag s1_pair --eval-T 5; run stage1_preserve.pair_dump --tag s1_pair16 --eval-T 5
run stage1_preserve.probes s1_pair_x4 s1_pair16_x4
echo "PAIR DONE $(date)"
