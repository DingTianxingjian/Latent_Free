#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 32 --mem=200G
#SBATCH -t 40:00:00
#SBATCH -J acdc_p3
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/p3_%j.out
# next_task.md section 5 on Newton: post-train ONE phase-2 arm on one LIBERO suite at LIBERO_TRAIN_FRAC % demos, then
# evaluate it closed loop (tasks 0-4 on GPU 0, 5-9 on GPU 1, 50 trials each) and merge -- workstation_run.sh plus
# --init_from_run, same recipe (2 GPUs x 8 x accum 4 = 64/step, 15k steps, flow head, 4 query tokens).
#
#   LATENT=tc_cont     INIT=acdc_la/ckpt/vla_runs/<p2_ours run>   SUITE=libero_object NOTE=p2ours_libero_object_frac10 \
#     sbatch acdc_la/scripts/run_vla_post2_newton.sh
#   LATENT=univla_cont INIT=acdc_la/ckpt/vla_runs/<p2_univla run> SUITE=libero_10     NOTE=p2univla_libero_10_frac10 ...
# UniVLA's LIBERO budget (global batch 128, 30k steps, Long 40k): NGPU=8 GA=2 on highgpu, or GA=8 on 2 GPUs.
# Rerunning the same command resumes training (--resume True) and skips it once the merged checkpoint exists.
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free
export PYTHONPATH=$PWD:$PWD/third_party/UniVLA WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false TORCH_HOME=$PWD/cache/torch \
       PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export LIBERO_TRAIN_FRAC=${FRAC:-10} CTRL_STAGGER_SEC=${CTRL_STAGGER_SEC:-90} CTRL_REITER_BATCHES=${CTRL_REITER_BATCHES:-2500}
[ "${FRAC:-10}" = full ] && unset LIBERO_TRAIN_FRAC     # FRAC=full: UniVLA's own split, train[:95%] (rlds/dataset.py)
PY=/lustre/fs1/home/ti874297/lf/envs/univla_vla/bin/python
source /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/scripts/gpu_preflight_newton.sh
L=${LATENT:?set LATENT}; S=${SUITE:?set SUITE}; N=${NOTE:?set NOTE}; DS=${S}_no_noops
X="--head flow --query_tokens ${QT:-4}"; [ "$L" = tc_cont ] && X="$X --tc_tag ${TC_TAG:-latp2_inv}"
[ -n "${INIT:-}" ] && X="$X --init_from_run $INIT"          # unset INIT = the no-phase-2 control (section 6)
RUN=$(ls -d acdc_la/ckpt/vla_runs/${L}+${DS}+b*+*s${STEPS:-15000}*--${N} 2>/dev/null | head -1)
if [ -z "$RUN" ] || [ ! -f "$RUN/config.json" ]; then          # config.json appears with the merged checkpoint
  $PY -m torch.distributed.run --standalone --nproc-per-node ${NGPU:-2} -m acdc_la.downstream.vla.finetune_libero_ctrl --latent $L $X \
      --dataset_name $DS --run_id_note $N --batch_size ${BS:-8} --grad_accumulation_steps ${GA:-4} --max_steps ${STEPS:-15000} --save_steps ${SAVE:-1000} \
      --resume True 2>&1 \
    | grep --line-buffered -vE "xFormers|warnings.warn|UserWarning|tensorflow/|I0000|E0000|W0000|oneDNN|TF-TRT|cuda_|AtomicFunction|NoneType|Exception ignored|atomic_function.py"
  RC=${PIPESTATUS[0]}; echo "POST2 train $L $N exit $RC $(date)"; [ $RC = 0 ] || exit $RC
  RUN=$(ls -d acdc_la/ckpt/vla_runs/${L}+${DS}+b*+*s${STEPS:-15000}*--${N} | head -1)
fi
[ "${SKIP_EVAL:-0}" = 1 ] && { echo "[p3] SKIP_EVAL=1: training only"; exit 0; }
echo "[p3] evaluating $RUN"
export PYTHONPATH=$PYTHONPATH:/lustre/fs1/home/ti874297/lf/envs/LIBERO MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
NG=${NGPU:-2}                                                  # the 10 tasks spread over every GPU of the job
for g in $(seq 0 $((NG - 1))); do
  T=$(seq $g $NG 9 | paste -sd_)
  CUDA_VISIBLE_DEVICES=$g LIBERO_TASK_IDS=$T $PY -m acdc_la.downstream.vla.run_libero_eval_cont --pretrained_checkpoint $RUN \
      --task_suite_name $S --num_trials_per_task ${TRIALS:-50} --run_id_note nw_$N --local_log_dir acdc_la/results/vla_eval \
      > acdc_la/logs/p3_eval_${N}_gpu$g.log 2>&1 < /dev/null &
done
wait
$PY -m acdc_la.downstream.vla.eval_merge --suite $S --note nw_$N --kind cont
echo "POST2 done $L $N $(date)"
