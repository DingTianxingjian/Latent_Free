#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 8 --mem=110G
#SBATCH -t 02:00:00
#SBATCH -J acdc_tcgate
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/tcgate_%j.out
# Gate before spending 8-11 h of VLA on a tokenizer: OXE probes (from the training dump) + LIBERO oracle action L1.
# Decision rule (review 2026-09-15): tc_cont enters the VLA if oracle L1 < 0.344 and the temporal probes do not regress;
# tc_vq256 enters only if its oracle L1 is close to the continuous one (post-hoc PQ was 0.401).
set -u -o pipefail
cd /anvil/scratch/x-tding2/latent_free
source /anvil/scratch/x-tding2/latent_free/acdc_la/scripts/common.sh
TAGS=${TAGS:-"tc_cont tc_vq256"}
run stage1_preserve.probes --code-only $TAGS s1_acdc_1ep s1_mean_1ep
export PYTHONPATH=/anvil/scratch/x-tding2/latent_free:/anvil/scratch/x-tding2/envs/LIBERO
/anvil/scratch/x-tding2/envs/libero/bin/python -u -m acdc_la.downstream.libero_offline --tags $TAGS s1_acdc_1ep --steps 2000 --no-decoder --out tc_gate 2>&1 | grep --line-buffered -vE "xFormers|warnings.warn|Using cache|UserWarning|Gym has|gymnasium|migration"
echo "TC GATE DONE $(date)"
