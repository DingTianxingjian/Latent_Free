#!/bin/bash
# One line per node: does the GPU allocation Slurm hands out actually work?  (Newton has H100 nodes where nvidia-smi
# finds no device or fewer devices than allocated, and torch then fails with "CUDA unknown error".)
N=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU")
T=$(timeout 120 /lustre/fs1/home/ti874297/lf/envs/univla_vla/bin/python -c "
import torch
try:
    n=torch.cuda.device_count(); assert torch.cuda.is_available()
    for i in range(n): torch.zeros(8,device=f'cuda:{i}').sum().item()
    print(f'torch_ok={n}')
except Exception as e: print('torch_FAIL')" 2>/dev/null | tail -1)
echo "$(hostname -s) alloc=${CUDA_VISIBLE_DEVICES:-none} smi=$N ${T:-torch_TIMEOUT}"
