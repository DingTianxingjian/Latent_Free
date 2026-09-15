#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 8 --mem=110G
#SBATCH -t 01:30:00
#SBATCH -J acdc_bits
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/bits_%j.out
# Bits of its own latent label that each trained VLA recovers at deployment -- the instrument that replaces the
# cross-method column of common/predictability.py (retraction 2026-09-15).  --split val is the RLDS validation
# split, which the LoRA fine-tune (train=True) never saw; --split train is the in-sample number that the 0.82 / 0.41
# training token accuracies correspond to.  Report the pair.
set -u -o pipefail
cd /anvil/scratch/x-tding2/latent_free
export PYTHONPATH=/anvil/scratch/x-tding2/latent_free:/anvil/scratch/x-tding2/latent_free/third_party/UniVLA:/anvil/scratch/x-tding2/envs/LIBERO
export TORCH_HOME=/anvil/scratch/x-tding2/latent_free/cache/torch PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python WANDB_MODE=disabled
PY=/anvil/scratch/x-tding2/envs/univla_vla/bin/python
for R in ${RUNS:?set RUNS}; do
  for D in ${SUITES:-libero_spatial_no_noops}; do
   for SP in ${SPLITS:-val train}; do
    $PY -u -m acdc_la.downstream.vla.token_bits --run $R --dataset $D --split $SP --n ${N:-600} 2>&1 \
      | grep --line-buffered -vE "xFormers|warnings.warn|UserWarning|tensorflow/|I0000|E0000|W0000|oneDNN|TF-TRT|cuda_|Gym has|gymnasium|migration"
    echo "BITS $R $D $SP exit $?"
   done
  done
done
echo "BITS DONE $(date)"
