#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=highgpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH -N 1 -n 32 --mem=900G
#SBATCH -t 01:00:00
#SBATCH -J p2_fft_smoke
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/p2_fft_smoke_%j.out
# Smoke of FULL-PARAMETER phase 2 (--full_ft True, FSDP): ours (tc_cont / latp2_inv), acdc_oxe29 (GNM weights = UniVLA's
# omni_magic_soup_plus), latent-only (--lam_a 0), flow head, 4 query tokens.  Checks: model shards, steps run, loss is
# finite, the full checkpoint saves; records GPU memory, host RAM and s/step.  STEPS / BS / GA / NGPU are overridable.
cd /lustre/fs1/home/ti874297/lf/latent_free
export PYTHONPATH=$PWD:$PWD/third_party/UniVLA WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
       PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TORCH_HOME=$PWD/cache/torch TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
       UNIVLA_FIXED_WINDOW=1 CTRL_REITER_BATCHES=2500 CTRL_OBS_KEEP=${CTRL_OBS_KEEP-0,1,3,4,6,7,9,10,12,13}   # frames latp2_inv reads: T=K+1=5, stride 3, ro in {0,1} (test_obs_keep.py)
PY=/lustre/fs1/home/ti874297/lf/envs/univla_vla/bin/python
source acdc_la/scripts/gpu_preflight_newton.sh
J=${SLURM_JOB_ID:-local}
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 20 > acdc_la/logs/p2_fft_smoke_gpu_$J.csv &
G=$!
( while true; do echo "$(date +%T) rss=$(ps -u $USER -o rss= | awk '{s+=$1} END {print int(s/1048576)}')G"; sleep 30; done > acdc_la/logs/p2_fft_smoke_ram_$J.log ) &
R=$!
NN=${SLURM_NNODES:-1}
if [ "$NN" -gt 1 ]; then      # multi-node: one torchrun per node, c10d rendezvous on the first node, NCCL over InfiniBand
  export MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
  LAUNCH="srun --ntasks-per-node=1 --kill-on-bad-exit=1 $PY -m torch.distributed.run --nnodes $NN --nproc-per-node ${NGPU:-2} --rdzv_backend c10d --rdzv_endpoint $MASTER:29511 --rdzv_id $J"
else
  LAUNCH="$PY -m torch.distributed.run --standalone --nproc-per-node ${NGPU:-4}"
fi
F="xFormers|warnings.warn|UserWarning|tensorflow/|I0000|E0000|W0000|oneDNN|TF-TRT|cuda_|AtomicFunction|NoneType|Exception ignored|atomic_function.py"
ARGS="--latent tc_cont --tc_tag latp2_inv --head flow --query_tokens 4 --lam_a 0 --merge False --full_ft True \
      --data_root_dir $PWD/data/oxe_raw --dataset_name acdc_oxe29 --batch_size ${BS:-8} --grad_accumulation_steps ${GA:-1} \
      --max_steps ${STEPS:-30} --save_steps ${SAVE:-10} --log_every 5 --run_id_note p2fft_smoke_$J"
echo "===== run 1: train ${STEPS:-30} steps, sharded checkpoint every ${SAVE:-10}, full weights at the end"
$LAUNCH -m acdc_la.downstream.vla.finetune_libero_ctrl $ARGS 2>&1 | grep --line-buffered -vE "$F"
RC=${PIPESTATUS[0]}; echo "===== run 1 exit $RC"
if [ $RC = 0 ]; then
  echo "===== run 2: --resume True from the step-${STEPS:-30} sharded checkpoint (one extra step, no save)"
  $LAUNCH -m acdc_la.downstream.vla.finetune_libero_ctrl $ARGS --resume True 2>&1 | grep --line-buffered -vE "$F"
  RC=${PIPESTATUS[0]}; echo "===== run 2 exit $RC"
fi
kill $G $R 2>/dev/null
echo "FFT_SMOKE exit $RC  peak GPU MiB: $(sort -t, -k2 -n acdc_la/logs/p2_fft_smoke_gpu_$J.csv | tail -1)  peak RSS: $(sort -t= -k2 -n acdc_la/logs/p2_fft_smoke_ram_$J.log | tail -1) $(date)"
exit $RC
