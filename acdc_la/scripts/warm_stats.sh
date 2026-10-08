#!/bin/bash
cd /home/ti874297/Projects/latent_free
export PYTHONPATH=$PWD:$PWD/third_party/UniVLA WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
       PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 UNIVLA_FIXED_WINDOW=1
( while true; do echo "$(date -u +%H:%M:%S) rss=$(ps -eo rss,args | grep '[w]arm_stats.py' | awk '{s+=$1} END {print int(s/1024/1024)}')G avail=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo)/1024/1024 ))G"; sleep 60; done > logs_migrate/warm_stats_mem.log ) &
MON=$!
CUDA_VISIBLE_DEVICES= envs/univla_vla/bin/python acdc_la/scripts/warm_stats.py
echo "WARM_STATS exit $? $(date -u)"; kill $MON 2>/dev/null
