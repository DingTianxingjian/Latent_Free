#!/bin/bash
# next_task.md section 4: phase-2 latent-action pretraining of ONE arm on the full acdc_oxe29 mixture, on this
# workstation (2 GPUs x micro-batch 8 x accum 4 = the prescribed global batch 64).  acdc_la/scripts/run_vla_phase2.sh
# is the Slurm twin for Anvil's arm C.
#
# The data root MUST be absolute: the RLDS statistics cache key is sha256(str(builder.info) + state_obs_keys +
# standardize_fn source) and builder.info carries the data_dir spelling verbatim, so "data/oxe_raw" and
# "$PWD/data/oxe_raw" are different keys.  acdc_la/scripts/warm_stats.sh cached all 29 sources (4 h 05 m, peak RSS 39 GB)
# under the absolute spelling; with the relative one 20 sources miss and BOTH ranks recompute them -- that is what
# OOM-killed the first smoke on 2026-10-08.
#
#   LATENT=tc_cont TC_TAG=latp2_inv NOTE=p2_ours    bash acdc_la/scripts/run_vla_phase2_ws.sh
#   LATENT=univla_cont              NOTE=p2_univla  bash acdc_la/scripts/run_vla_phase2_ws.sh
#   RESUME=1 <same vars>                            # continue a killed run; pass the SAME STEPS or exp_id changes
set -u -o pipefail
cd /home/ti874297/Projects/latent_free
export PYTHONPATH=$PWD:$PWD/third_party/UniVLA WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
       PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TORCH_HOME=$PWD/cache/torch TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
       UNIVLA_FIXED_WINDOW=1 CTRL_REITER_BATCHES=${CTRL_REITER_BATCHES:-2500}
L=${LATENT:?set LATENT=tc_cont|univla_cont}; N=${NOTE:?set NOTE}
X="--head flow --query_tokens ${QT:-4}"; [ "$L" = tc_cont ] && X="$X --tc_tag ${TC_TAG:-latp2_inv}"
[ "${RESUME:-0}" = 1 ] && X="$X --resume True"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 60 > logs_migrate/p2_${N}_mem.csv &
M=$!
envs/univla_vla/bin/python -m torch.distributed.run --standalone --nproc-per-node ${NGPU:-2} \
    -m acdc_la.downstream.vla.finetune_libero_ctrl \
    --latent $L $X --lam_a 0 --merge False \
    --data_root_dir $PWD/data/oxe_raw --dataset_name ${MIX:-acdc_oxe29} \
    --batch_size ${BS:-8} --grad_accumulation_steps ${GA:-4} \
    --max_steps ${STEPS:-30000} --save_steps ${SAVE:-1000} --run_id_note $N ${EXTRA:-} 2>&1 \
  | grep --line-buffered -vE "xFormers|warnings.warn|UserWarning|tensorflow/|I0000|E0000|W0000|oneDNN|TF-TRT|cuda_|AtomicFunction|NoneType|Exception ignored|^Traceback|atomic_function.py"
RC=${PIPESTATUS[0]}; kill $M 2>/dev/null
echo "PHASE2 $L $N exit $RC  peak GPU MiB: $(sort -t, -k2 -n logs_migrate/p2_${N}_mem.csv | tail -1) $(date -u)"
exit $RC
