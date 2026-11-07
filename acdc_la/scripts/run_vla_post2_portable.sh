#!/bin/bash
# Post-train a phase-2 run on one LIBERO suite on any machine without Slurm (2026-11-05): run_vla_post2_newton.sh with the
# Newton paths out, for the full-parameter 20k model on the UCF workstation.  Full split and UniVLA's budget by default:
# global batch 128, Long 40k steps, the others 30k.  Trains, merges, evaluates 500 episodes over GPUS, merges the eval.
# Rerunning resumes training and skips it once the merged checkpoint exists; a crashed trainer is retried from its last save.
#
#   PY=envs/univla_vla/bin/python INIT=<p2fft_ours run dir> SUITE=libero_10 NOTE=p2fft20k_libero_10_full GPUS="0 1" \
#     RUNROOT=/data/.../vla_runs bash acdc_la/scripts/run_vla_post2_portable.sh
#
# Real-robot (non-LIBERO) data takes DS, the data root through EXTRA, and no LIBERO eval:
#   ... DS=piper_pick_can EXTRA="--data_root_dir $PWD/data/real_rlds" SKIP_EVAL=1 GLOBAL=64 STEPS=10000 NOTE=terra_piper ...
# Unset LIBERO_TRAIN_FRAC (FRAC=full, the default) is UniVLA's train[:95%], so the held-out side is train[95%:].
#
# BS x GA x number of GPUs must stay at the arm's global batch (GLOBAL, default 128), or the result is not comparable
# with the runs it is being compared against -- the full-split Newton runs are 128 (b32x4), the 10 % cells 64 (b32x2),
# so a 10 % replicate is GLOBAL=64 FRAC=10 STEPS=15000.  The iterator restart is
# counted in micro-batches but the RLDS leak goes with samples (~250 MB / 1000 samples per rank), so it is derived from BS:
# 20000 samples per iterator, ~5 GB per rank, whatever BS is.  A fixed 2500 at BS=32 would be ~20 GB per rank.
set -u -o pipefail
cd "$(dirname "$0")/../.."
PY=${PY:?python of the env}; INIT=${INIT:?phase-2 run dir}; N=${NOTE:?}; G=(${GPUS:-0 1}); NG=${#G[@]}
# DS names the RLDS dataset directly, for data that is not a LIBERO suite; then SUITE is only the eval suite and may be
# omitted when the eval is skipped.  Unset DS keeps the old behaviour exactly: the suite name plus _no_noops.
DS=${DS:-${SUITE:?suite name, or DS for a non-LIBERO dataset}_no_noops}; S=${SUITE:-$DS}
GB=${GLOBAL:-128}                              # the global batch to hold: 128 = UniVLA's full-split budget, 64 = the 10 % cells
BS=${BS:-8}; GA=${GA:-$((GB / NG / BS))}; [ $((BS * GA * NG)) = "$GB" ] || { echo "BS x GA x GPUs = $((BS * GA * NG)), not $GB"; exit 1; }
[ "$S" = libero_10 ] && STEPS=${STEPS:-40000} || STEPS=${STEPS:-30000}
ROOT=${RUNROOT:-acdc_la/ckpt/vla_runs}; mkdir -p "$ROOT" acdc_la/logs acdc_la/results/vla_eval
[ -f "$INIT/vla_full.pt" ] || { echo "$INIT has no vla_full.pt"; exit 1; }
export PYTHONPATH=$PWD:$PWD/third_party/UniVLA WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false TORCH_HOME=$PWD/cache/torch \
       PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
       CTRL_STAGGER_SEC=${CTRL_STAGGER_SEC:-90} CTRL_REITER_BATCHES=${CTRL_REITER_BATCHES:-$((20000 / BS))}
[ "${FRAC:-full}" = full ] && unset LIBERO_TRAIN_FRAC || export LIBERO_TRAIN_FRAC=$FRAC
LOG=acdc_la/logs/post2_${N}.log
find_run() { ls -d "$ROOT"/tc_cont+${DS}+b*+*s${STEPS}*--${N} 2>/dev/null | head -1; }
RUN=$(find_run)
if [ -z "$RUN" ] || [ ! -f "$RUN/config.json" ]; then          # config.json appears with the merged checkpoint
  for try in 1 2 3; do
    echo "[post2] $N train, attempt $try, BS $BS x GA $GA x $NG GPUs, reiter $CTRL_REITER_BATCHES $(date)" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=$(echo ${G[*]} | tr ' ' ,) $PY -m torch.distributed.run --nnodes 1 --master_addr 127.0.0.1 --master_port $((15000 + RANDOM % 30000)) \
        --nproc-per-node $NG -m acdc_la.downstream.vla.finetune_libero_ctrl --latent tc_cont --head flow --query_tokens 4 --tc_tag latp2_inv \
        --init_from_run $INIT --dataset_name $DS --run_id_note $N --run_root_dir $ROOT --batch_size $BS --grad_accumulation_steps $GA \
        --max_steps $STEPS --save_steps ${SAVE:-1000} --resume True ${EXTRA:-} >> $LOG 2>&1 < /dev/null
    RC=$?; echo "[post2] $N train exit $RC $(date)" | tee -a $LOG; [ $RC = 0 ] && break; sleep 60
  done
  [ $RC = 0 ] || { echo "[post2] $N giving up"; exit $RC; }
  RUN=$(find_run)
fi
[ "${SKIP_EVAL:-0}" = 1 ] && exit 0
export PYTHONPATH=$PYTHONPATH:${LIBERO:-$PWD/envs/LIBERO} MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
for i in "${!G[@]}"; do
  T=$(seq $i $NG 9 | paste -sd_)
  CUDA_VISIBLE_DEVICES=${G[$i]} LIBERO_TASK_IDS=$T $PY -m acdc_la.downstream.vla.run_libero_eval_cont --pretrained_checkpoint $RUN \
      --task_suite_name $S --num_trials_per_task ${TRIALS:-50} --run_id_note ws_$N --local_log_dir acdc_la/results/vla_eval \
      > acdc_la/logs/post2_eval_${N}_gpu${G[$i]}.log 2>&1 < /dev/null &
done
wait
$PY -m acdc_la.downstream.vla.eval_merge --suite $S --note ws_$N --kind cont
echo "[post2] done $N $RUN $(date)"
