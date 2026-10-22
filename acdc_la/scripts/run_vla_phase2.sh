#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=highgpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH -N 1 -n 32 --mem=400G
#SBATCH -t 08:00:00
#SBATCH -J acdc_p2
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/p2_%j.out
# PHASE 2 -- latent-action pretraining of the policy on the cross-embodiment mixture (UniVLA's second stage, which our
# 2026-09-19/20 comparison skipped).  Latent prediction ONLY (--lam_a 0): those sources have no common action space,
# and a latent that the action head can infer from (o_t, l) cannot pay for itself downstream -- that is exactly what
# the no-latent control measured on Goal (443/500 vs 434/500 and 447/500, ledger 2026-09-20).
#
#   LATENT=tc_cont TC_TAG=latp2_inv NOTE=p2_ours     sbatch acdc_la/scripts/run_vla_phase2.sh
#   LATENT=univla_cont              NOTE=p2_univla   sbatch acdc_la/scripts/run_vla_phase2.sh
#   RESUME=1 <same vars>            sbatch ...       # walltime chunk 2, 3, ... (SAME STEPS, or exp_id changes)
#
# STEPS (default 30000) x 64 samples = 1.9M clips, ~1/100 of their pretraining: a MATCHED-budget comparison between
# the two latents, not a reproduction of UniVLA's scale.  Post-train each arm with run_vla_post2.sh.
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free
export PYTHONPATH=/lustre/fs1/home/ti874297/lf/latent_free:/lustre/fs1/home/ti874297/lf/latent_free/third_party/UniVLA \
       WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
       TORCH_HOME=/lustre/fs1/home/ti874297/lf/latent_free/cache/torch TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
       UNIVLA_FIXED_WINDOW=1      # both labellers need a 14-frame window on every source (patch in rlds/dataset.py)
PY=/lustre/fs1/home/ti874297/lf/envs/univla_vla/bin/python
source /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/scripts/gpu_preflight_newton.sh
L=${LATENT:?set LATENT=tc_cont|univla_cont}; N=${NOTE:?set NOTE}
X="--head flow --query_tokens ${QT:-4}"; [ "$L" = tc_cont ] && X="$X --tc_tag ${TC_TAG:-latp2_inv}"
[ "${RESUME:-0}" = 1 ] && X="$X --resume True"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader -l 60 > acdc_la/logs/p2_mem_${SLURM_JOB_ID:-local}_${N}.csv &
MEMPID=$!
$PY -m torch.distributed.run --standalone --nproc-per-node ${NGPU:-4} -m acdc_la.downstream.vla.finetune_libero_ctrl \
    --latent $L $X --lam_a 0 --merge False \
    --data_root_dir ${DATA:-/lustre/fs1/home/ti874297/lf/latent_free/data/oxe_raw} \
    --dataset_name ${MIX:-acdc_oxe29} \
    --max_steps ${STEPS:-30000} --save_steps ${SAVE:-1000} --run_id_note $N ${EXTRA:-} 2>&1 \
  | grep --line-buffered -vE "xFormers|warnings.warn|UserWarning|tensorflow/|I0000|E0000|W0000|oneDNN|TF-TRT|cuda_|AtomicFunction|NoneType|Exception ignored|^Traceback|atomic_function.py"
RC=${PIPESTATUS[0]}; kill $MEMPID 2>/dev/null
echo "PHASE2 $L $N exit $RC  peak GPU memory (MiB): $(sort -t, -k2 -n acdc_la/logs/p2_mem_${SLURM_JOB_ID:-local}_${N}.csv | tail -1) $(date)"
exit $RC
