#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 16 --mem=120G
#SBATCH -t 04:00:00
#SBATCH -J acdc_s4
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/s4_%j.out
# Stage IV: portable core + reconstruction residual on top of the 1-epoch LATP checkpoint, plus the "same extra decoder
# steps" control (LATP continued on the full code), then the three existing evaluations on both.
source /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/scripts/common.sh
STEPS=${STEPS:-10000}; INIT=${INIT:-s3_latp_1ep}; TAG=${TAG:-s4_pcrr}; CTRL=${CTRL:-s3_latp_1ep_cont}
run stage4_portable.train --tag smoke_s4 --init-from $INIT --steps 40 --warmup 10 --limit-eps 200 --val-per-source 20 --workers 8 --beta-u 2e-4 || exit 1
run stage4_portable.train --tag $TAG --init-from $INIT --steps $STEPS --workers 14 --beta-u 2e-4 ${EXTRA:-}
run stage3_transport.train --tag $CTRL --init-from $INIT --steps $STEPS --workers 14
run stage1_preserve.probes --code-only $TAG $CTRL $INIT s1_acdc_1ep
run stage2_disentangle.probes --no-ref $TAG $CTRL $INIT
run stage3_transport.probes $TAG $CTRL $INIT
echo "STAGE4 DONE $(date)"
