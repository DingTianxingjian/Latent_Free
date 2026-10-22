#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH -N 1 -n 1 -c 8 --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J fetch_w
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/fetch_weights_%j.out
# Public weights the runs read from cache/: OpenVLA-7B base (15 GB) and the two DINOv2 torch.hub entries (REPRODUCE.md §0).
cd /lustre/fs1/home/ti874297/lf/latent_free
E=/lustre/fs1/home/ti874297/lf/envs
HF_HOME=/lustre/fs1/home/ti874297/lf/hf_cache $E/univla_vla/bin/python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('openvla/openvla-7b', local_dir='cache/vla/openvla-7b'))"
echo "OPENVLA exit $?"
TORCH_HOME=$PWD/cache/torch $E/latent_free/bin/python -c "
import torch
for n in ('dinov2_vitb14','dinov2_vitb14_reg'): torch.hub.load('facebookresearch/dinov2', n); print('ok', n)"
echo "DINO exit $?"
