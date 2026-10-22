#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 32 --mem=490G
#SBATCH -t 60:00:00
#SBATCH -J acdc_p2
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/p2_%j.out
# PHASE 2 on Newton -- the workstation recipe of next_task.md section 4 (2 GPUs x micro-batch 8 x accum 4 = global
# batch 64), run in memory-bounded segments.
#
#   LATENT=tc_cont TC_TAG=latp2_inv NOTE=p2_ours     sbatch acdc_la/scripts/run_vla_phase2_newton.sh
#   LATENT=univla_cont              NOTE=p2_univla   sbatch acdc_la/scripts/run_vla_phase2_newton.sh
#
# Why segments: on the full acdc_oxe29 mixture the RLDS pipeline's host memory grows ~3 GB/min for 2 ranks
# (214 GB at 15 min -> 444 GB at 90 min, jobs 899991/899992, 2026-10-12) and CTRL_REITER_BATCHES does not return it,
# so a single process is OOM-killed near step 2000 on a 502 GB node.  This wrapper watches the job's RSS; once it
# passes SOFT_G it waits for the next checkpoint (trainer_state.pt rewritten) and restarts the trainer with
# --resume True; past HARD_G it restarts at once (losing the steps since the last checkpoint).  Every segment passes
# the SAME --max_steps, so the exp_id / run directory is unchanged.  A rerun of this script resumes too.
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free
export PYTHONPATH=$PWD:$PWD/third_party/UniVLA WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
       PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TORCH_HOME=$PWD/cache/torch TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
       UNIVLA_FIXED_WINDOW=1 CTRL_REITER_BATCHES=${CTRL_REITER_BATCHES:-2500}
PY=/lustre/fs1/home/ti874297/lf/envs/univla_vla/bin/python
source /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/scripts/gpu_preflight_newton.sh
L=${LATENT:?set LATENT=tc_cont|univla_cont}; N=${NOTE:?set NOTE}
STEPS=${STEPS:-30000}; SOFT_G=${SOFT_G:-380}; HARD_G=${HARD_G:-465}
X="--head flow --query_tokens ${QT:-4}"; [ "$L" = tc_cont ] && X="$X --tc_tag ${TC_TAG:-latp2_inv}"
JID=${SLURM_JOB_ID:-local}
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 60 > acdc_la/logs/p2_mem_${JID}_${N}.csv &
MEMPID=$!
rss_g() { ps -u $USER -o rss= | awk '{s+=$1} END {print int(s/1048576)}'; }
state_mtime() { stat -c %Y acdc_la/ckpt/vla_runs/${L}+*+s${STEPS}+*--${N}/trainer_state.pt 2>/dev/null | sort -n | tail -1; }
SEG=0; RC=1
while [ $SEG -lt 80 ]; do
  SEG=$((SEG + 1)); echo "[seg $SEG] start $(date) last checkpoint mtime=$(state_mtime)"
  setsid $PY -m torch.distributed.run --standalone --nproc-per-node 2 -m acdc_la.downstream.vla.finetune_libero_ctrl \
      --latent $L $X --lam_a 0 --merge False --resume True \
      --data_root_dir $PWD/data/oxe_raw --dataset_name ${MIX:-acdc_oxe29} \
      --batch_size 8 --grad_accumulation_steps 4 \
      --max_steps $STEPS --save_steps ${SAVE:-500} --run_id_note $N ${EXTRA:-} \
      > >(grep --line-buffered -vE "xFormers|warnings.warn|UserWarning|tensorflow/|I0000|E0000|W0000|oneDNN|TF-TRT|cuda_|AtomicFunction|NoneType|Exception ignored|^Traceback|atomic_function.py") 2>&1 &
  PID=$!; MARK=""; RESTART=0
  while kill -0 $PID 2>/dev/null; do
    sleep 30; R=$(rss_g)
    echo "$(date +%F_%T) seg=$SEG rss=${R}G" >> acdc_la/logs/p2_rss_${JID}_${N}.log
    if [ -z "$MARK" ] && [ "$R" -ge "$SOFT_G" ]; then MARK=$(date +%s); echo "[seg $SEG] rss ${R}G >= ${SOFT_G}G, restarting at the next checkpoint"; fi
    if [ -n "$MARK" ] && [ "$(state_mtime)" -gt "$MARK" ] 2>/dev/null && [ $(( $(date +%s) - $(state_mtime) )) -ge 60 ]; then RESTART=1; echo "[seg $SEG] checkpoint written, restarting (rss ${R}G) $(date)"; fi
    if [ "$R" -ge "$HARD_G" ]; then RESTART=1; echo "[seg $SEG] rss ${R}G >= ${HARD_G}G HARD limit, restarting WITHOUT a fresh checkpoint $(date)"; fi
    if [ $RESTART = 1 ]; then kill -TERM -- -$PID 2>/dev/null; sleep 20; kill -KILL -- -$PID 2>/dev/null; break; fi
  done
  wait $PID; RC=$?
  if [ $RESTART = 1 ]; then pkill -KILL -u $USER -f finetune_libero_ctrl; sleep 20; continue; fi
  break            # the trainer ended by itself: finished (0) or failed
done
kill $MEMPID 2>/dev/null
echo "PHASE2 $L $N exit $RC after $SEG segment(s)  peak GPU MiB: $(sort -t, -k2 -n acdc_la/logs/p2_mem_${JID}_${N}.csv | tail -1) $(date)"
exit $RC
