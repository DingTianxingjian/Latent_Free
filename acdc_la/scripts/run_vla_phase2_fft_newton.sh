#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=highgpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:8
#SBATCH --exclude=evc104
#SBATCH -N 1 -n 1 -c 112 --mem=950G
#SBATCH -t 5-00:00:00
#SBATCH -J p2_fft
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/p2_fft_%j.out
# FULL-PARAMETER phase 2 at UniVLA's scale (2026-10-17): the whole 7B trains with FSDP on the 29-source acdc_oxe29 mixture
# (= UniVLA's omni_magic_soup_plus), latent-only (--lam_a 0), labels from latp2_inv (ledger 2026-10-12 -> 10-17).
# UniVLA's stage-2 recipe (prismatic/conf/vla.py, Exp_DinoSigLIP_224px_OXE_Magic_Soup_Plus): global batch 1024,
# 20k steps, lr 2e-5 constant, weight decay 0, grad-norm clip 1 -> --univla_optim True.  Here NGPU x BS x GA = 1024.
#
#   sbatch acdc_la/scripts/run_vla_phase2_fft_newton.sh                       # 8 GPUs x 16 x accum 8
#   NGPU=4 BS=16 GA=16 sbatch --gres=gpu:nvidia_h100_80gb_hbm3:4 -c 56 --mem=600G ...   # same global batch on 4 GPUs
#   LR-decay branch from the finished constant-lr run (2026-10-29): 2k more steps, lr linear 2e-5 -> 0, own run dir
#   STEPS=22000 NOTE=p2fft_ours_decay EXTRA="--branch_from <p2fft_ours run dir> --decay_start 20000" NGPU=4 ... sbatch ...
#
# Memory-bounded segments, as run_vla_phase2_newton.sh: host RSS of the RLDS pipeline grows for every rank; once it
# passes SOFT_G the trainer is restarted with --resume at the next sharded checkpoint (written to a temp dir and swapped
# in, trainer_state.pt last); past HARD_G at once.  The exp_id does not change across segments.  Rerunning resumes.
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free
export PYTHONPATH=$PWD:$PWD/third_party/UniVLA WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
       PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TORCH_HOME=$PWD/cache/torch TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
       UNIVLA_FIXED_WINDOW=1 CTRL_REITER_BATCHES=${CTRL_REITER_BATCHES:-2500} \
       CTRL_OBS_KEEP=${CTRL_OBS_KEEP-0,1,3,4,6,7,9,10,12,13}     # the frames latp2_inv reads (acdc_la/scripts/test_obs_keep.py)
PY=/lustre/fs1/home/ti874297/lf/envs/univla_vla/bin/python
source /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/scripts/gpu_preflight_newton.sh
NGPU=${NGPU:-8}; BS=${BS:-16}; GA=${GA:-8}; STEPS=${STEPS:-20000}; N=${NOTE:-p2fft_ours}
SOFT_G=${SOFT_G:-700}; HARD_G=${HARD_G:-880}
# A continuation queued with --dependency=afterany may start after the run already finished: then do nothing.
if ls acdc_la/ckpt/vla_runs/tc_cont+acdc_oxe29+*+fft+s${STEPS}+*--${N}/vla_full.pt >/dev/null 2>&1; then
  echo "PHASE2-FFT $N already finished (vla_full.pt exists), nothing to do $(date)"; exit 0; fi
JID=${SLURM_JOB_ID:-local}
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -l 60 > acdc_la/logs/p2_fft_gpu_${JID}.csv &
MEMPID=$!
rss_g() { ps -u $USER -o rss= | awk '{s+=$1} END {print int(s/1048576)}'; }
state_mtime() { stat -c %Y acdc_la/ckpt/vla_runs/tc_cont+acdc_oxe29+*+fft+s${STEPS}+*--${N}/trainer_state.pt 2>/dev/null | sort -n | tail -1; }
SEG=0; RC=1; FAILS=0; CRASH_CKPT=""
while [ $SEG -lt 200 ]; do
  SEG=$((SEG + 1)); echo "[seg $SEG] start $(date) last checkpoint mtime=$(state_mtime)"
  setsid $PY -m torch.distributed.run --standalone --nproc-per-node $NGPU -m acdc_la.downstream.vla.finetune_libero_ctrl \
      --latent tc_cont --tc_tag ${TC_TAG:-latp2_inv} --head flow --query_tokens 4 --lam_a 0 --merge False \
      --full_ft True --univla_optim True --learning_rate ${LR:-2e-5} \
      --data_root_dir $PWD/data/oxe_raw --dataset_name ${MIX:-acdc_oxe29} \
      --batch_size $BS --grad_accumulation_steps $GA --max_steps $STEPS --save_steps ${SAVE:-500} \
      --log_every ${LOGE:-10} --resume True --run_id_note $N ${EXTRA:-} \
      > >(grep --line-buffered -vE "^# |xFormers|warnings.warn|UserWarning|tensorflow/|I0000|E0000|W0000|oneDNN|TF-TRT|cuda_|AtomicFunction|NoneType|Exception ignored|atomic_function.py") 2>&1 &
  PID=$!; MARK=""; RESTART=0
  while kill -0 $PID 2>/dev/null; do
    sleep 30; R=$(rss_g)
    echo "$(date +%F_%T) seg=$SEG rss=${R}G" >> acdc_la/logs/p2_fft_rss_${JID}.log
    if [ -z "$MARK" ] && [ "$R" -ge "$SOFT_G" ]; then MARK=$(date +%s); echo "[seg $SEG] rss ${R}G >= ${SOFT_G}G, restarting at the next checkpoint"; fi
    if [ -n "$MARK" ] && [ "$(state_mtime)" -gt "$MARK" ] 2>/dev/null && [ $(( $(date +%s) - $(state_mtime) )) -ge 60 ]; then RESTART=1; echo "[seg $SEG] checkpoint written, restarting (rss ${R}G) $(date)"; fi
    if [ "$R" -ge "$HARD_G" ]; then RESTART=1; echo "[seg $SEG] rss ${R}G >= ${HARD_G}G HARD limit, restarting WITHOUT a fresh checkpoint $(date)"; fi
    if [ $RESTART = 1 ]; then kill -TERM -- -$PID 2>/dev/null; sleep 30; kill -KILL -- -$PID 2>/dev/null; break; fi
  done
  wait $PID; RC=$?
  if [ $RESTART = 1 ]; then pkill -KILL -u $USER -f finetune_libero_ctrl; sleep 30; continue; fi
  [ $RC = 0 ] && break            # the trainer finished
  # Crashed (2026-10-28: segment 21 died inside the dcp resume with a collective mismatch, and the job ended and gave
  # up its node).  Retry from the last checkpoint in the same allocation; give up after 3 crashes with no new checkpoint.
  [ "$(state_mtime)" != "$CRASH_CKPT" ] && FAILS=0
  FAILS=$((FAILS + 1)); CRASH_CKPT=$(state_mtime)
  if [ $FAILS -ge 3 ]; then echo "[seg $SEG] trainer crashed (exit $RC) 3 times on the same checkpoint, giving up $(date)"; break; fi
  echo "[seg $SEG] trainer crashed (exit $RC), retry $FAILS of 2 from the last checkpoint $(date)"
  pkill -KILL -u $USER -f finetune_libero_ctrl; sleep 60
done
kill $MEMPID 2>/dev/null
echo "PHASE2-FFT $N exit $RC after $SEG segment(s) $(date)"
exit $RC
