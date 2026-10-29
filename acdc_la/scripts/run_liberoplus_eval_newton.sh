#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 1 -c 16 --mem=120G
#SBATCH -t 24:00:00
#SBATCH -J liberoplus
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/liberoplus_%j.out
# LIBERO-Plus (CVPR 2026; github sylvestf/LIBERO-plus, assets HF Sylvest/LIBERO-plus) robustness evaluation of a
# post-trained run: same evaluator as LIBERO (run_libero_eval_cont), 1 trial per task as the benchmark prescribes; each
# suite holds ~2,500 perturbed tasks, split into NSHARD shards; this job runs the shards listed in SHARDS (one per GPU).
# Per-category numbers: acdc_la/downstream/vla/liberoplus_summary.py.
#   RUN=<run dir> SUITE=libero_spatial NOTE=<tag> NSHARD=8 SHARDS="0 1" sbatch acdc_la/scripts/run_liberoplus_eval_newton.sh
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free
PY=/lustre/fs1/home/ti874297/lf/envs/univla_vla/bin/python
source acdc_la/scripts/gpu_preflight_newton.sh
export PYTHONPATH=$PWD:$PWD/third_party/UniVLA:/lustre/fs1/home/ti874297/lf/envs/LIBERO-plus WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
       TORCH_HOME=$PWD/cache/torch PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
       MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH=/lustre/fs1/home/ti874297/lf/envs/liberoplus_cfg
R=${RUN:?}; S=${SUITE:?}; N=${NOTE:?}; NS=${NSHARD:?}; g=0
for k in ${SHARDS:?}; do
  CUDA_VISIBLE_DEVICES=$g LIBERO_TASK_SHARD="$k/$NS" $PY -m acdc_la.downstream.vla.run_libero_eval_cont --pretrained_checkpoint $R \
      --task_suite_name $S --num_trials_per_task 1 --run_id_note plus_$N --local_log_dir acdc_la/results/vla_eval_plus \
      > acdc_la/logs/liberoplus_${N}_${S}_s${k}of${NS}.log 2>&1 < /dev/null &
  g=$((g + 1))
done
wait
echo "LIBERO-Plus done $N $S shards [$SHARDS]/$NS $(date)"
