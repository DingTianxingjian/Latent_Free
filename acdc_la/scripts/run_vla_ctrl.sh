#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=highgpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH -N 1 -n 32 --mem=400G
#SBATCH -t 08:00:00
#SBATCH -J acdc_vla
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/vla_%j.out
# Controlled UniVLA LIBERO fine-tuning, one arm per job: LATENT=univla|acdc, STEPS (default 15000), CODEBOOK for the acdc arm.
# 4 x H100, batch 8 x accum 2 per GPU = 64 samples/step.  Writes ckpt/vla_runs/<exp_id>/ (merged model, action_decoder.pt, ctrl_meta.json).
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free
export PYTHONPATH=/lustre/fs1/home/ti874297/lf/latent_free:/lustre/fs1/home/ti874297/lf/latent_free/third_party/UniVLA WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false TORCH_HOME=/lustre/fs1/home/ti874297/lf/latent_free/cache/torch PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
PY=/lustre/fs1/home/ti874297/lf/envs/univla_vla/bin/python
L=${LATENT:?set LATENT=univla|acdc}
$PY -m torch.distributed.run --standalone --nproc-per-node ${NGPU:-4} -m acdc_la.downstream.vla.finetune_libero_ctrl --latent $L --max_steps ${STEPS:-15000} --save_steps ${SAVE:-5000} --acdc_codebook ${CODEBOOK:-vq_s1_acdc_1ep_K64.npz} ${EXTRA:-} 2>&1 | grep --line-buffered -vE "xFormers|warnings.warn|UserWarning|tensorflow/|I0000|E0000|W0000|oneDNN|TF-TRT|cuda_|AtomicFunction|NoneType|Exception ignored|^Traceback|atomic_function.py"
echo "VLA CTRL $L exit ${PIPESTATUS[0]} $(date)"
