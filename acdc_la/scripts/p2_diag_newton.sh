#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 1 -c 16 --mem=440G
#SBATCH -t 04:00:00
#SBATCH -J p2_diag
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/p2_diag_%j.out
# Diagnosis of "full-parameter phase 2 at step 5000 post-trains to 57.6 % on 10 % Long, the 1/10 LoRA to 61.6 %"
# (2026-10-29).  Each GPU runs one arm through the same stages, so the two arms see identical procedures:
#   zs_lib : lr 0 on the LIBERO-Long 10 % post-training data, log every update -> the phase-2 model's latent R2 on
#            LIBERO BEFORE any post-training (the logged step-25 value was 0.16 fft vs 0.66 LoRA)
#   ft_lib : the first 40 post-training updates at the real lr (3.5e-4, batch 64 as 2 GPUs x 8 x 4) -> the trajectory
#   heldout: prior_gate on the FINAL post-trained policy, held-out split train[95%:]: latent R2 + sampled action L1
#   zs_oxe : lr 0 on the phase-2 mixture (acdc_oxe29, phase-2 env) -> each phase-2 model's own objective, same
#            evaluator; for fft this also validates the consolidated checkpoint (training log: R2 ~0.82 at 5k)
# and a CPU side job: weight change vs openvla-7b per component + heads key check (p2_diag_weights.py).
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free
export PYTHONPATH=$PWD:$PWD/third_party/UniVLA WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false TORCH_HOME=$PWD/cache/torch \
       PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CTRL_REITER_BATCHES=2500
PY=/lustre/fs1/home/ti874297/lf/envs/univla_vla/bin/python
source acdc_la/scripts/gpu_preflight_newton.sh
V=acdc_la/ckpt/vla_runs; D=acdc_la/ckpt/diag_runs; O=acdc_la/results/p2_diag; mkdir -p $D $O
P2_FFT=$V/p2fft_ours_step5000
P2_LORA=$V/tc_cont+acdc_oxe29+b32x2+lr0.00035+r32+s30000+fm+q4--p2_ours
PT_FFT=$V/tc_cont+libero_10_no_noops+b32x2+lr0.00035+r32+s15000+fm+q4--p2fft5000_libero_10_frac10
PT_LORA=$V/tc_cont+libero_10_no_noops+b32x2+lr0.00035+r32+s15000+fm+q4--p2ours_libero_10_frac10
FILT="xFormers|warnings.warn|UserWarning|tensorflow/|I0000|E0000|W0000|oneDNN|TF-TRT|cuda_|AtomicFunction|NoneType|Exception ignored|atomic_function.py"

ctrl() {   # gpu arm init stage extra-args...
  local g=$1 arm=$2 init=$3 st=$4; shift 4
  CUDA_VISIBLE_DEVICES=$g $PY -m torch.distributed.run --nnodes 1 --master_addr 127.0.0.1 --master_port $((15000 + RANDOM % 30000)) --nproc-per-node 1 -m acdc_la.downstream.vla.finetune_libero_ctrl \
      --latent tc_cont --tc_tag latp2_inv --head flow --query_tokens 4 --init_from_run $init --merge False \
      --run_id_note diag_${st}_${arm} --log_every 1 --save_steps 100000 "$@" 2>&1 | grep --line-buffered -vE "$FILT"
}
want() { [[ " ${STAGES:-zs_lib ft_lib heldout zs_oxe} " == *" $1 "* ]]; }   # STAGES="..." runs a subset
arm() {   # gpu arm p2run posttrained-run
  local g=$1 arm=$2 p2=$3 pt=$4
  if want zs_lib; then echo "[diag] $arm zs_lib $(date)"
  ( export LIBERO_TRAIN_FRAC=10; ctrl $g $arm $p2 zs_lib --dataset_name libero_10_no_noops --batch_size 8 --grad_accumulation_steps 4 \
      --max_steps 25 --learning_rate 0.0 ) > $O/zs_lib_$arm.log; fi
  if want ft_lib; then echo "[diag] $arm ft_lib $(date)"
  ( export LIBERO_TRAIN_FRAC=10; ctrl $g $arm $p2 ft_lib --dataset_name libero_10_no_noops --batch_size 8 --grad_accumulation_steps 8 \
      --max_steps 50 --learning_rate 3.5e-4 ) > $O/ft_lib_$arm.log; fi
  if want heldout; then echo "[diag] $arm heldout $(date)"
  CUDA_VISIBLE_DEVICES=$g $PY -m acdc_la.downstream.vla.prior_gate --tag latp2_inv --suite libero_10 --n 2000 --no_prior --vla $pt \
      --out $O/heldout_$arm.json 2>&1 | grep -vE "$FILT" > $O/heldout_$arm.log; fi
  if want zs_oxe; then echo "[diag] $arm zs_oxe $(date)"
  ( export UNIVLA_FIXED_WINDOW=1 CTRL_OBS_KEEP=0,1,3,4,6,7,9,10,12,13; ctrl $g $arm $p2 zs_oxe --data_root_dir $PWD/data/oxe_raw \
      --dataset_name acdc_oxe29 --lam_a 0 --batch_size 8 --grad_accumulation_steps 4 --max_steps 25 --learning_rate 0.0 \
      --shuffle_buffer_size 4000 ) > $O/zs_oxe_$arm.log; fi
  echo "[diag] $arm done $(date)"
}
# ARMS="fft lora" (default, one GPU each) or one arm per 1-GPU job: ARMS=fft sbatch --gres=gpu:nvidia_h100_pcie:1 ...
g=0
for A in ${ARMS:-fft lora}; do
  if [ $A = fft ] && [ "${WEIGHTS:-1}" = 1 ]; then
    $PY acdc_la/scripts/p2_diag_weights.py --fft $P2_FFT --lora_adapter $V/adapters/$(basename $P2_LORA) --lora_run $P2_LORA \
        --out $O/weights.json > $O/weights.log 2>&1 &
    arm $g fft $P2_FFT $PT_FFT &
  elif [ $A = fft ]; then
    arm $g fft $P2_FFT $PT_FFT &
  else
    arm $g lora $P2_LORA $PT_LORA &
  fi
  g=$((g + 1)); sleep 120                       # stagger the 7B loads
done
wait
for f in $(ls $O/*.log | grep -E "$(echo ${ARMS:-fft lora} | tr ' ' '|')|weights"); do echo "== $f"; grep -E "^\[ctrl:tc_cont\]|r2_vla|act_l1|Traceback|Error|\[diag\]" $f | cut -c1-200 | tail -60; done
echo "P2-DIAG done $(date)"
