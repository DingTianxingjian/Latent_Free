#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH -N 1 -n 1 -c 16 --mem=96G
#SBATCH -t 12:00:00
#SBATCH -J warm_stats
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/warm_stats_%j.out
# Newton twin of warm_stats.sh: compute the acdc_oxe29 RLDS statistics once, in one process, CPU only (ledger 2026-10-08).
# The cache key contains the data-root spelling: every later run must pass the same absolute path
# (/lustre/fs1/home/ti874297/lf/latent_free/data/oxe_raw, what acdc_la.paths.LF gives here).
cd /lustre/fs1/home/ti874297/lf/latent_free
export PYTHONPATH=$PWD:$PWD/third_party/UniVLA WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
       PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 UNIVLA_FIXED_WINDOW=1
CUDA_VISIBLE_DEVICES= /lustre/fs1/home/ti874297/lf/envs/univla_vla/bin/python -u acdc_la/scripts/warm_stats.py
echo "WARM_STATS exit $? $(date -u)"
