#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 1 -c 32 --mem=300G
#SBATCH -t 04:00:00
#SBATCH -J p2_smoke
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/p2_smoke_%j.out
# Newton twin of p2_smoke_mix.sh -- next_task.md section 3: full acdc_oxe29 mixture smoke, both arms, 20 steps, the
# workstation recipe (2 GPUs x micro-batch 8 x accum 4).  Run warm_stats_newton.sh first; the data root is ABSOLUTE
# and must keep this spelling (statistics cache key, ledger 2026-10-08).
cd /lustre/fs1/home/ti874297/lf/latent_free
export PYTHONPATH=$PWD:$PWD/third_party/UniVLA WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
       PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TORCH_HOME=$PWD/cache/torch TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
       UNIVLA_FIXED_WINDOW=1 CTRL_REITER_BATCHES=2500
PY=/lustre/fs1/home/ti874297/lf/envs/univla_vla/bin/python
source /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/scripts/gpu_preflight_newton.sh
MEM=acdc_la/logs/p2_smoke_mem_${SLURM_JOB_ID:-local}.csv
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 30 > $MEM &
M=$!
( while true; do echo "$(date -u +%H:%M:%S) host_used=$(( ($(awk '/MemTotal/{print $2}' /proc/meminfo)-$(awk '/MemAvailable/{print $2}' /proc/meminfo))/1024/1024 ))G job_rss=$(ps -u $USER -o rss= | awk '{s+=$1} END {print int(s/1024/1024)}')G"; sleep 60; done > acdc_la/logs/p2_smoke_ram_${SLURM_JOB_ID:-local}.log ) &
R=$!
for ARM in "tc_cont --tc_tag latp2_inv" "univla_cont"; do
  N=$(echo $ARM | cut -d' ' -f1); echo "===== SMOKE $N start $(date -u)"
  $PY -m torch.distributed.run --standalone --nproc-per-node 2 -m acdc_la.downstream.vla.finetune_libero_ctrl \
      --latent $ARM --head flow --query_tokens 4 --lam_a 0 --merge False \
      --data_root_dir $PWD/data/oxe_raw --dataset_name acdc_oxe29 \
      --batch_size 8 --grad_accumulation_steps 4 --max_steps 20 --save_steps 1000 --log_every 5 --run_id_note p2smoke 2>&1 \
    | grep --line-buffered -vE "xFormers|warnings.warn|UserWarning|tensorflow/|I0000|E0000|W0000|oneDNN|TF-TRT|cuda_|AtomicFunction|Exception ignored|atomic_function.py"
  echo "===== SMOKE $N exit ${PIPESTATUS[0]} $(date -u)"
done
kill $M $R 2>/dev/null; echo "peak GPU MiB: $(sort -t, -k2 -n $MEM | tail -1)"
