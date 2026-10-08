#!/bin/bash
# next_task.md section 3: full acdc_oxe29 mixture smoke, both arms, 20 steps. First run also computes the RLDS
# dataset statistics for the 9 uncached big sources (a full pass over their records) -- that dominates the wall time.
# NOTE: the data root must be ABSOLUTE -- the statistics cache key is sha256(str(builder.info)), which contains the
# data_dir spelling verbatim.  warm_stats.sh cached the 29 sources under the absolute path; "data/oxe_raw" misses 20
# of them and both DDP ranks recompute them, which is what OOM-killed this smoke on 2026-10-08.
cd /home/ti874297/Projects/latent_free
export PYTHONPATH=$PWD:$PWD/third_party/UniVLA WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
       PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TORCH_HOME=$PWD/cache/torch TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
       UNIVLA_FIXED_WINDOW=1 CTRL_REITER_BATCHES=2500
PY=envs/univla_vla/bin/python
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 30 > logs_migrate/p2_smoke_mix_mem.csv &
M=$!
for ARM in "tc_cont --tc_tag latp2_inv" "univla_cont"; do
  N=$(echo $ARM | cut -d' ' -f1); echo "===== SMOKE $N start $(date -u)"
  $PY -m torch.distributed.run --standalone --nproc-per-node 2 -m acdc_la.downstream.vla.finetune_libero_ctrl \
      --latent $ARM --head flow --query_tokens 4 --lam_a 0 --merge False \
      --data_root_dir $PWD/data/oxe_raw --dataset_name acdc_oxe29 \
      --batch_size 8 --grad_accumulation_steps 4 --max_steps 20 --save_steps 1000 --log_every 5 --run_id_note p2smoke
  echo "===== SMOKE $N exit $? $(date -u)"
done
kill $M 2>/dev/null; echo "peak GPU MiB: $(sort -t, -k2 -n logs_migrate/p2_smoke_mix_mem.csv | tail -1)"
