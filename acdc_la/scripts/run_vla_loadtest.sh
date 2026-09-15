#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 4 --mem=100G
#SBATCH -t 00:30:00
#SBATCH -J acdc_loadtest
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/loadtest_%j.out
# Cheap gate before a 3-hour evaluation: can the evaluator load each merged checkpoint and take one action?
set -u -o pipefail
cd /anvil/scratch/x-tding2/latent_free
export PYTHONPATH=/anvil/scratch/x-tding2/latent_free:/anvil/scratch/x-tding2/latent_free/third_party/UniVLA:/anvil/scratch/x-tding2/envs/LIBERO WANDB_MODE=disabled TORCH_HOME=/anvil/scratch/x-tding2/latent_free/cache/torch PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
PY=/anvil/scratch/x-tding2/envs/univla_vla/bin/python
for R in ${RUNS:?set RUNS}; do
  D=/anvil/scratch/x-tding2/latent_free/acdc_la/ckpt/vla_runs/$R
  MOD=acdc_la.downstream.vla.run_libero_eval_ctrl; grep -q '"latent": "acdc_cont"' $D/ctrl_meta.json && MOD=acdc_la.downstream.vla.run_libero_eval_cont
  echo "== $R via $MOD"
  RAW=/anvil/scratch/x-tding2/latent_free/acdc_la/logs/loadtest_raw_${SLURM_JOB_ID}_${R:0:6}.txt
  echo n | timeout 900 $PY -m $MOD --pretrained_checkpoint $D --task_suite_name libero_spatial --num_trials_per_task 1 > $RAW 2>&1
  echo "LOADTEST $R exit $?  (full output: $RAW)"
  grep -vE "xFormers|warnings.warn|UserWarning|tensorflow/|I0000|E0000|W0000|oneDNN|TF-TRT|cuda_|AtomicFunction|NoneType|Exception ignored|atomic_function|robosuite WARNING|macros|Gym has|gymnasium|migration|^\s*$" $RAW | tail -30
done
echo "LOADTEST DONE $(date)"
