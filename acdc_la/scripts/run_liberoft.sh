#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=preemptable
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:1
#SBATCH --exclude=evc11
#SBATCH -N 1 -n 1 -c 8 --mem=64G
#SBATCH -t 8:00:00
#SBATCH -J liberoft
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/liberoft_%j.out
# LIBERO fine-tune of the Table-4 arms for the physical-replay test (phys_replay.py).  Both arms continue their own
# eatabm_<arm>_s<SEED> checkpoint with their own objective on the four libero_sim_* caches (demos 8-27; 0-7 held out),
# identical data / steps / lr / seed.  SEED=0 sbatch acdc_la/scripts/run_liberoft.sh   (preemption-safe: --resume)
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free
source /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/scripts/common.sh
export TORCH_HOME=$PWD/cache/torch
S=${SEED:?}
scontrol update JobId=$SLURM_JOB_ID JobName=liberoft_s$S 2>/dev/null
run stage3_transport.train_multi --tag liberoft --arms ${ARMS:-full,recon} --finetune-from eatabm --init-from tc_cont \
    --sources libero_sim_spatial libero_sim_object libero_sim_goal libero_sim_10 \
    --steps ${STEPS:-4000} --warmup 1 --batch 64 --lr ${LR:-1e-4} --seed $S --workers ${WORKERS:-12} --save-every 500 --eval-every 0 --resume
echo "LIBEROFT s$S done $(date)"
