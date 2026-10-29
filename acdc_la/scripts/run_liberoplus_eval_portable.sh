#!/bin/bash
# LIBERO-Plus evaluation on any machine with NVIDIA GPUs (2026-10-29): the Newton launcher without Slurm or Newton paths,
# for running a suite off Newton (the Long suite while Newton's monthly GPU quota is exhausted).
#
# Needs on the machine:
#   - this repo (branch newton), run from its root; third_party/UniVLA present (experiments/robot + prismatic)
#   - a Python env like Newton's univla_vla: torch 2.2, transformers 4.40-4.44, tensorflow (CPU is fine), robosuite /
#     mujoco / bddl as LIBERO-plus requires
#   - LIBERO-plus (github sylvestf/LIBERO-plus) with assets.zip unzipped into libero/libero (HF Sylvest/LIBERO-plus)
#   - the post-trained run dir (merged model + action_decoder.pt + dataset_statistics.json)
# Shards are dealt round-robin over GPUS; each GPU runs its shards one after another (one 7B policy per GPU, ~20 GB).
# Results: acdc_la/results/vla_eval_plus/cont_<suite>_plus_<NOTE>_s<k>of<NSHARD>.json (copy back to Newton, then
# `python -m acdc_la.downstream.vla.liberoplus_summary --note <NOTE>`); rerunning skips finished shards and resumes
# unfinished ones task by task.
#
#   PY=~/envs/univla/bin/python LIBERO_PLUS=~/LIBERO-plus RUN=~/ckpt/<run dir> SUITE=libero_10 NOTE=p2ours_full \
#   NSHARD=8 SHARDS="0 1 2 3 4 5 6 7" GPUS="0 1" bash acdc_la/scripts/run_liberoplus_eval_portable.sh
set -u -o pipefail
cd "$(dirname "$0")/../.."
PY=${PY:?python of the env}; LP=$(cd "${LIBERO_PLUS:?LIBERO-plus repo}" && pwd); R=${RUN:?run dir}
S=${SUITE:?}; N=${NOTE:?}; NS=${NSHARD:?}; SH=(${SHARDS:?}); G=(${GPUS:-0})
[ -d "$LP/libero/libero/assets" ] || { echo "no $LP/libero/libero/assets: unzip LIBERO-plus assets.zip there"; exit 1; }
[ -f "$R/action_decoder.pt" ] && [ -f "$R/config.json" ] || { echo "$R is not a merged post-trained run dir"; exit 1; }
CFG=$(mktemp -d); trap 'rm -rf $CFG' EXIT            # LIBERO reads its paths from $LIBERO_CONFIG_PATH/config.yaml
cat > $CFG/config.yaml <<EOF
benchmark_root: $LP/libero/libero
bddl_files: $LP/libero/libero/bddl_files
init_states: $LP/libero/libero/init_files
datasets: $LP/datasets
assets: $LP/libero/libero/assets
EOF
export PYTHONPATH=$PWD:$PWD/third_party/UniVLA:$LP WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
       PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
       LIBERO_CONFIG_PATH=$CFG LIBERO_RESUME=1           # an interrupted shard continues where it stopped
OUT=acdc_la/results/vla_eval_plus; mkdir -p $OUT acdc_la/logs
worker() {   # gpu shard...
  local g=$1; shift
  for k in "$@"; do
    f=$OUT/cont_${S}_plus_${N}_s${k}of${NS}.json
    if [ -f $f ] && $PY -c "import json,sys; d=json.load(open('$f')); sys.exit(len(d['per_task_id']) < len(d['task_ids']))"; then
      echo "[plus] shard $k done already, skipping"; continue; fi
    echo "[plus] GPU $g shard $k/$NS start $(date)"
    CUDA_VISIBLE_DEVICES=$g LIBERO_TASK_SHARD="$k/$NS" $PY -m acdc_la.downstream.vla.run_libero_eval_cont --pretrained_checkpoint $R \
        --task_suite_name $S --num_trials_per_task 1 --run_id_note plus_$N --local_log_dir $OUT \
        > acdc_la/logs/liberoplus_${N}_${S}_s${k}of${NS}.log 2>&1 < /dev/null
    echo "[plus] GPU $g shard $k/$NS exit $? $(date)"
  done
}
for i in "${!G[@]}"; do
  mine=(); for j in "${!SH[@]}"; do [ $((j % ${#G[@]})) = $i ] && mine+=(${SH[$j]}); done
  [ ${#mine[@]} -gt 0 ] && worker ${G[$i]} "${mine[@]}" &
done
wait
echo "LIBERO-Plus done $N $S shards [${SH[*]}]/$NS $(date)"
