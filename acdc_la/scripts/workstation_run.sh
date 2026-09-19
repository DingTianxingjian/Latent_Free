#!/bin/bash
# Workstation (no Slurm, 2 GPUs): train ONE continuous-arm VLA cell at 10 % demos, evaluate it on both GPUs, merge.
#   LATENT=tc_cont SUITE=libero_goal NOTE=latp2inv_fmq4_frac10 bash acdc_la/scripts/workstation_run.sh
# LATENT: tc_cont (ours, TC_TAG default latp2_inv) | univla_cont (UniVLA's latent, needs claude_survey/ckpt/s2mix_1ep.pt) | none
# SUITE: libero_spatial | libero_object | libero_goal | libero_10.   QT (query tokens, default 4), FRAC (default 10).
# Batch 8 x accum 4 x 2 GPUs = 64/step, the Anvil budget (run dir reads b32x2 instead of b16x4).  ~13 h train, ~1.5-3 h eval.
set -euo pipefail
cd "$(dirname "$0")/../.."                                              # latent_free/
PY=${PY:-envs/univla_vla/bin/python}
export PYTHONPATH=$PWD:$PWD/third_party/UniVLA WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false TORCH_HOME=$PWD/cache/torch \
       PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export LIBERO_TRAIN_FRAC=${FRAC:-10} CTRL_STAGGER_SEC=${CTRL_STAGGER_SEC:-90}
: "${LATENT:?set LATENT}" "${SUITE:?set SUITE}" "${NOTE:?set NOTE}"
DS=${SUITE}_no_noops
EXTRA="--head flow --query_tokens ${QT:-4}"; [ "$LATENT" = tc_cont ] && EXTRA="$EXTRA --tc_tag ${TC_TAG:-latp2_inv}"
mkdir -p acdc_la/logs
if [ "${SKIP_TRAIN:-0}" != 1 ]; then
  $PY -m torch.distributed.run --standalone --nproc-per-node 2 -m acdc_la.downstream.vla.finetune_libero_ctrl --latent $LATENT $EXTRA \
      --dataset_name $DS --run_id_note $NOTE --grad_accumulation_steps 4 --max_steps 15000 2>&1 | tee acdc_la/logs/ws_train_${NOTE}_${SUITE}.log
fi
RUN=$(ls -d acdc_la/ckpt/vla_runs/${LATENT}+${DS}+b32x2+*--${NOTE} | head -1); echo "[ws] evaluating $RUN"
export PYTHONPATH=$PYTHONPATH:$PWD/envs/LIBERO MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
for g in 0 1; do
  T=$([ $g = 0 ] && echo 0_1_2_3_4 || echo 5_6_7_8_9)
  CUDA_VISIBLE_DEVICES=$g LIBERO_TASK_IDS=$T $PY -m acdc_la.downstream.vla.run_libero_eval_cont --pretrained_checkpoint $RUN \
      --task_suite_name $SUITE --num_trials_per_task 50 --run_id_note ws_$NOTE --local_log_dir acdc_la/results/vla_eval \
      > acdc_la/logs/ws_eval_${NOTE}_${SUITE}_gpu$g.log 2>&1 &
done
wait
$PY -m acdc_la.downstream.vla.eval_merge --suite $SUITE --note ws_$NOTE --kind cont
echo "[ws] done: commit acdc_la/results/vla_eval/*ws_${NOTE}* and $RUN/train_log.jsonl so Anvil sees the result"
