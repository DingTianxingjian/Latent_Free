#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 8 --mem=60G
#SBATCH -t 01:30:00
#SBATCH -J univla_matched
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/univla_matched_%j.out
# Budget-matched UniVLA stage-2 (s2mix_1ep: their recipe, our data, 128k x 64): dump on the val clips + Stage I/II probes +
# LIBERO offline accuracy, so it sits in every table next to s1_acdc_1ep / s1_mean_1ep.
source /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/scripts/common.sh
UARM=${UARM:-s2mix_1ep}; TAG=${TAG:-univla_matched}; REF=${REF:-"s1_acdc_1ep s1_mean_1ep"}
run baselines.univla --arm $UARM --tag $TAG --workers 8 || exit 1
run stage1_preserve.probes --code-only $TAG univla_official $REF
run stage2_disentangle.probes --no-ref $TAG univla_official $REF
echo "UNIVLA MATCHED DONE $(date)"
