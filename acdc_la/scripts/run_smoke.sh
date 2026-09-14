#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 8 --mem=60G
#SBATCH -t 00:40:00
#SBATCH -J acdc_smoke
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/smoke_%j.out
# Plumbing test of the whole repo on the GPU: the three trainers (tiny), the legacy-init path, every probe.  Not a result.
source /anvil/scratch/x-tding2/latent_free/acdc_la/scripts/common.sh
COMMON="--limit-eps 200 --val-per-source 20 --batch 16 --steps 60 --warmup 20 --workers 8"
run stage1_preserve.train --tag smoke_s1 $COMMON || exit 1
run stage1_preserve.train --tag smoke_s1mean --pool mean $COMMON || exit 1
run stage2_disentangle.train --tag smoke_s2 --init-from smoke_s1 $COMMON || exit 1
run stage3_transport.train --tag smoke_s3 --init-from smoke_s1 $COMMON || exit 1
run stage3_transport.train --tag smoke_s3legacy --init-from rec_src $COMMON || exit 1
run baselines.univla --tag smoke_univla --limit-eps 200 --val-per-source 20 --batch 16 --workers 8 || exit 1
run stage1_preserve.probes smoke_s1 smoke_s1mean smoke_univla
run stage2_disentangle.probes smoke_s2 smoke_univla
run stage3_transport.probes smoke_s3 smoke_s3legacy; run stage3_transport.recovery smoke_s3
echo "SMOKE DONE $(date)"
