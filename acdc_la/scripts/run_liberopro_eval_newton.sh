#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 1 -c 16 --mem=120G
#SBATCH -t 24:00:00
#SBATCH -J liberopro
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/liberopro_%j.out
# LIBERO-PRO (arXiv 2510.03827, github Zxy-MLlab/LIBERO-PRO @ eafdb80) robustness evaluation of a post-trained run:
# the same evaluator as LIBERO (run_libero_eval_cont, center crop, seed 7, base-suite max steps), on a perturbed suite
# libero_{spatial,object,goal,10}_{lan,object,swap,task}.  LIBERO-PRO's bddl/init files come from HF zhouxueyang/LIBERO-Pro;
# its own config dir keeps the standard LIBERO config untouched.
#   RUN=acdc_la/ckpt/vla_runs/<run> SUITE=libero_goal_swap NOTE=<tag> [TRIALS=50] sbatch acdc_la/scripts/run_liberopro_eval_newton.sh
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free
PY=/lustre/fs1/home/ti874297/lf/envs/univla_vla/bin/python
source acdc_la/scripts/gpu_preflight_newton.sh
export PYTHONPATH=$PWD:$PWD/third_party/UniVLA:/lustre/fs1/home/ti874297/lf/envs/LIBERO-PRO WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false \
       TORCH_HOME=$PWD/cache/torch PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
       MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH=/lustre/fs1/home/ti874297/lf/envs/libero_pro_cfg
R=${RUN:?set RUN}; S=${SUITE:?set SUITE}; N=${NOTE:?set NOTE}; NG=${NGPU:-2}
for g in $(seq 0 $((NG - 1))); do
  T=$(seq $g $NG 9 | paste -sd_)
  CUDA_VISIBLE_DEVICES=$g LIBERO_TASK_IDS=$T $PY -m acdc_la.downstream.vla.run_libero_eval_cont --pretrained_checkpoint $R \
      --task_suite_name $S --num_trials_per_task ${TRIALS:-50} --run_id_note pro_$N --local_log_dir acdc_la/results/vla_eval_pro \
      > acdc_la/logs/liberopro_${N}_${S}_gpu$g.log 2>&1 < /dev/null &
done
wait
$PY -m acdc_la.downstream.vla.eval_merge --suite $S --note pro_$N --kind cont --dir acdc_la/results/vla_eval_pro --trials ${TRIALS:-50}
echo "LIBERO-PRO done $N $S $(date)"
