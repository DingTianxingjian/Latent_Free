#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH -N 1 -n 1 -c 16 --mem=16G
#SBATCH -t 4-00:00:00
#SBATCH -J oxe_fetch
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/logs/oxe_fetch_%j.out
# Fetch the 29 acdc_oxe29 sources from the public OXE bucket into data/oxe_raw (a symlink into /lustre/fs1/groups/ytian).
# CPU only; idempotent -- resubmit after any interruption (complete shards are skipped by size).
cd /lustre/fs1/home/ti874297/lf/latent_free
/lustre/fs1/home/ti874297/lf/envs/latent_free/bin/python -u phase1/oxe/oxe_fetch.py --workers ${WORKERS:-24} "$@"
