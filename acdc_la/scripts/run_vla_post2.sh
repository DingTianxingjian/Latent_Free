#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=highgpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH -N 1 -n 32 --mem=400G
#SBATCH -t 08:00:00
#SBATCH -J acdc_p3
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/p3_%j.out
# POST-TRAINING of a phase-2 pretrained arm: identical to the workstation cells (LIBERO suite @ LIBERO_TRAIN_FRAC %,
# flow head + query tokens, 15k x 64) except that LoRA and the heads start from INIT (a run_vla_phase2.sh run dir).
# The matched no-phase-2 arms are the 2026-09-19/20 workstation cells; together they price UniVLA's second stage.
#
#   LATENT=tc_cont TC_TAG=latp2_inv INIT=acdc_la/ckpt/vla_runs/<p2 run> SUITE=libero_goal NOTE=p2ours_goal_frac10 \
#     sbatch acdc_la/scripts/run_vla_post2.sh
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free
export PYTHONPATH=/lustre/fs1/home/ti874297/lf/latent_free:/lustre/fs1/home/ti874297/lf/latent_free/third_party/UniVLA \
       WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
       TORCH_HOME=/lustre/fs1/home/ti874297/lf/latent_free/cache/torch TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export LIBERO_TRAIN_FRAC=${FRAC:-10}
PY=/lustre/fs1/home/ti874297/lf/envs/univla_vla/bin/python
L=${LATENT:?set LATENT}; N=${NOTE:?set NOTE}; I=${INIT:?set INIT=<phase-2 run dir>}; S=${SUITE:-libero_goal}
X="--head flow --query_tokens ${QT:-4}"; [ "$L" = tc_cont ] && X="$X --tc_tag ${TC_TAG:-latp2_inv}"
[ "${RESUME:-0}" = 1 ] && X="$X --resume True"
$PY -m torch.distributed.run --standalone --nproc-per-node ${NGPU:-4} -m acdc_la.downstream.vla.finetune_libero_ctrl \
    --latent $L $X --init_from_run $I --dataset_name ${S}_no_noops \
    --max_steps ${STEPS:-15000} --save_steps ${SAVE:-5000} --run_id_note $N ${EXTRA:-} 2>&1 \
  | grep --line-buffered -vE "xFormers|warnings.warn|UserWarning|tensorflow/|I0000|E0000|W0000|oneDNN|TF-TRT|cuda_|AtomicFunction|NoneType|Exception ignored|^Traceback|atomic_function.py"
echo "POST2 $L $N exit ${PIPESTATUS[0]} $(date)"
