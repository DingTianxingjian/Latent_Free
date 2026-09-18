#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 8 --mem=120G
#SBATCH -t 06:00:00
#SBATCH -J acdc_vlaeval
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/vlaeval_%j.out
# 2026-09-16: job 20754113 died with SIGABRT (exit 134) at the first rollout on h006 -- model loaded, benchmark
# built, then a native abort with no traceback, the signature of an EGL/MuJoCo context failure on that node.
# Successful evaluations ran on h005 and h008.  Pass --exclude=h004,h006 until it is understood.
# UniVLA's LIBERO evaluator on a controlled checkpoint: RUN=<run_dir under ckpt/vla_runs>, SUITE (libero_spatial), TRIALS (50).
# NOTE=<tag> names the output files (default: the first 40 characters of the run dir, which is NOT unique across arms).
# LIBERO_TASK_IDS=0_1 evaluates only those tasks, so a 500-episode evaluation can be split across short jobs that
# backfill; merge the parts with `python -m acdc_la.downstream.vla.eval_merge --suite ... --note ... --kind cont|token`.
set -u -o pipefail
cd /anvil/scratch/x-tding2/latent_free
export PYTHONPATH=/anvil/scratch/x-tding2/latent_free:/anvil/scratch/x-tding2/latent_free/third_party/UniVLA:/anvil/scratch/x-tding2/envs/LIBERO WANDB_MODE=disabled TORCH_HOME=/anvil/scratch/x-tding2/latent_free/cache/torch PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
PY=/anvil/scratch/x-tding2/envs/univla_vla/bin/python
R=/anvil/scratch/x-tding2/latent_free/acdc_la/ckpt/vla_runs/${RUN:?set RUN}
MOD=acdc_la.downstream.vla.run_libero_eval_ctrl; grep -qE '"latent": "(acdc_cont|tc_cont|none)"' $R/ctrl_meta.json && MOD=acdc_la.downstream.vla.run_libero_eval_cont
echo n | $PY -m $MOD --pretrained_checkpoint $R --task_suite_name ${SUITE:-libero_spatial} --num_trials_per_task ${TRIALS:-50} --local_log_dir /anvil/scratch/x-tding2/latent_free/acdc_la/results/vla_eval --run_id_note ${NOTE:-$(basename $R | cut -c1-40)} 2>&1 | grep --line-buffered -vE "xFormers|warnings.warn|UserWarning|tensorflow/|I0000|E0000|W0000|oneDNN|TF-TRT|cuda_|AtomicFunction|NoneType|Exception ignored|^Traceback|atomic_function.py|robosuite WARNING|macros"
RC=${PIPESTATUS[1]}
echo "VLA EVAL exit $RC $(date)"
# 2026-09-18: exit with the evaluator's status.  Before this the script ended on `echo`, so a MuJoCo/EGL abort (134)
# on a bad node was reported by Slurm as COMPLETED and a missing half of the tasks was invisible.
exit $RC
