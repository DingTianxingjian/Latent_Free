#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=preemptable
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc11,evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 1 -c 16 --mem=160G
#SBATCH -t 20:00:00
#SBATCH -J eatab
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/eatab_%j.out
# Rebuttal ablation A (2026-11-06): every arm +15k steps from the same 65k tc_cont checkpoint, 3 seeds, same data.
# The capped OXE caches the paper's rows were trained on (Anvil) are not on Newton, so EVERY arm, seed 0 included,
# is retrained here on the uncapped caches (ACDC_FULL_DATA=1); the paper rows are not mixed into this table.
# Hyper-parameters are those stored in ckpt/latp2_inv.pt and ckpt/tc_cont_plus15k.pt.  The preemptable partition
# CANCELS on preemption: resubmitting the same ARM/SEED resumes from the last save (--resume, --save-every 1000).
#   ARM=recon|noinv|full|a1|a2|a3|a4 SEED=0 sbatch acdc_la/scripts/run_eat_ablation_newton.sh
source /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/scripts/common.sh
export ACDC_FULL_DATA=1 TORCH_HOME=$PWD/cache/torch
timeout 120 $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "EATAB NO GPU on $(hostname) -- bad node, exiting"; exit 3; }   # evc11 (2026-11-06) allocated a GPU torch could not see
A=${ARM:?}; S=${SEED:?}; TAG=eatab_${A}_s${S}
COMMON="--tag $TAG --init-from tc_cont --steps 15000 --warmup 4000 --batch 64 --lr 3e-4 --seed $S --workers 12 --save-every 1000 --resume"
V2="stage3_transport.train_v2 $COMMON --eval-every 3000 --lam-recover 0"
case $A in
  recon) run stage2_tc.train $COMMON --eval-every 0 ;;                                   # L_self only (tc_cont_plus15k)
  noinv) run $V2 --lam-effect 1 --lam-real 1 --lam-rel 0.5 --lam-inv 0 ;;               # EAT without L_inv (latp2_anchor)
  full)  run $V2 --lam-effect 1 --lam-real 1 --lam-rel 0.5 --lam-inv 1 ;;               # full EAT (latp2_inv)
  a1)    run $V2 --lam-effect 0 --lam-real 0 --lam-rel 0   --lam-inv 1 ;;               # L_self + L_inv, no transport term
  a2)    run $V2 --lam-effect 0 --lam-real 1 --lam-rel 0.5 --lam-inv 1 ;;               # full minus L_effect
  a3)    run $V2 --lam-effect 1 --lam-real 1 --lam-rel 0   --lam-inv 1 ;;               # full minus L_rel
  a4)    run $V2 --lam-effect 1 --lam-real 0 --lam-rel 0.5 --lam-inv 1 ;;               # full minus L_dist (MMD)
  *) echo "unknown ARM $A"; exit 1 ;;
esac
echo "EATAB $TAG done $(date)"
