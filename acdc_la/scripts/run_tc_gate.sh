#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 8 --mem=110G
#SBATCH -t 02:00:00
#SBATCH -J acdc_tcgate
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/tcgate_%j.out
# Gate before spending 8-11 h of VLA on a tokenizer: OXE probes (from the training dump) + LIBERO oracle action L1.
# Decision rule (review 2026-09-15): tc_cont enters the VLA if oracle L1 < 0.344 and the temporal probes do not regress;
# tc_vq256 enters only if its oracle L1 is close to the continuous one (post-hoc PQ was 0.401).
# Extended 2026-09-15: also report P = R2(z|F_0), S = R2(action|z) and usable = P*S.  At equal oracle L1 prefer the
# arm with the higher usable; a candidate that beats the oracle gate but drops usable below 0.020 repeats the closed-loop null.
set -u -o pipefail
cd /lustre/fs1/home/ti874297/lf/latent_free
source /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/scripts/common.sh
TAGS=${TAGS:-"tc_cont tc_vq256"}
# 2026-09-15: the tokenizers were stopped at ~half budget (tc_cont 65k, tc_vq256 75k of 128k) because resuming to
# the full budget would not have finished before the deadline.  The validation dump is normally written at the end of
# training, so write it here from whatever checkpoint exists.
for T in $TAGS; do
  [ -f acdc_la/results/${T}_val.npz ] || run stage2_tc.train --tag $T --eval-only --workers 8 || exit 1
done
run stage1_preserve.probes --code-only $TAGS s1_acdc_1ep s1_mean_1ep
run common.predictability $TAGS s1_acdc_1ep s1_mean_1ep   # P, S, usable -- WITHIN-FAMILY only (retraction 2026-09-15):
                                                         # univla is deliberately absent, its deployed code is 4 indices,
                                                         # not this 2048-d embedding, and the cross-method number lives in
                                                         # downstream/vla/token_bits.py instead.
export PYTHONPATH=/lustre/fs1/home/ti874297/lf/latent_free:/lustre/fs1/home/ti874297/lf/envs/LIBERO
/lustre/fs1/home/ti874297/lf/envs/libero/bin/python -u -m acdc_la.downstream.libero_offline --tags $TAGS s1_acdc_1ep --steps 2000 --no-decoder --out tc_gate 2>&1 | grep --line-buffered -vE "xFormers|warnings.warn|Using cache|UserWarning|Gym has|gymnasium|migration"
echo "TC GATE DONE $(date)"
