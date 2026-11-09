#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=preemptable
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc11,evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 1 -c 16 --mem=160G
#SBATCH -t 2-00:00:00
#SBATCH -J eatabm
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/eatabm_%j.out
# Rebuttal ablation A, 2026-11-07: three Table-4 arms per job on ONE data stream (stage3_transport/train_multi.py).
# The batch, its DINO features and the L_inv nuisance draw are shared, which the CPU smoke measured as the larger part
# of a step (3.3-4.0 s vs 0.9-1.6 s per arm), so 3 arms cost ~1.5-2x one arm instead of 3x.
# PREEMPTABLE = cancelled on preemption: resubmit the same GROUP/SEED; --resume continues every arm from its last save.
#   GROUP=g1|g2|g3 SEED=0 sbatch acdc_la/scripts/run_eat_multi_newton.sh
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free
source /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/scripts/common.sh
export ACDC_FULL_DATA=1 TORCH_HOME=$PWD/cache/torch
# 120 s was too short: cold torch imports from lustre took >2 min on evc32/33 (jobs 1177725/28), which then trained fine.
timeout 900 $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; rc=$?
[ $rc = 124 ] && { echo "EATABM torch import timed out on $(hostname)"; exit 4; }
[ $rc != 0 ] && { echo "EATABM NO GPU on $(hostname) -- bad node, exiting"; exit 3; }
G=${GROUP:?}; S=${SEED:?}
case $G in
  g1) ARMS=recon,a1,full ;;          # L_self | L_self + L_inv | full EAT (TERRA)
  g2) ARMS=noinv,a2,a3 ;;            # EAT w/o L_inv | w/o L_effect | w/o L_rel
  g3) ARMS=a4,a5,a6 ;;               # w/o L_dist | L_effect -> head | L_effect -> cycle
  all) ARMS=recon,a1,full,noinv,a2,a3,a4,a5,a6 ;;   # one job per seed: the step is data-bound (~1.5 s), each arm adds ~0.2 s
  *) echo "unknown GROUP $G"; exit 1 ;;
esac
scontrol update JobId=$SLURM_JOB_ID JobName=eatabm_${G}_s$S 2>/dev/null
run stage3_transport.train_multi --tag eatabm --arms $ARMS --init-from tc_cont --steps 15000 --warmup 4000 --batch 64 \
    --lr 3e-4 --seed $S --workers ${WORKERS:-12} --save-every 500 --eval-every 0 --resume
echo "EATABM $G s$S done $(date)"
