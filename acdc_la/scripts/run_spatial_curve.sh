#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 4 --mem=100G
#SBATCH -t 01:30:00
#SBATCH -J acdc_spatial
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/spatial_%j.out
# What does the shared 256 -> 16 patch pooling destroy?  No tokenizer, no training: extract the patch grid once on a
# subset of validation clips, pool it to 16x16 / 8x8 / 4x4 / 2x2, build the same [N, A1] evidence at each resolution,
# equalise dimension by PCA, and read the action off it with the standard leave-one-source-out ridge.  The nuisance
# column repeats it under the moving-patch distractor.
source /anvil/scratch/x-tding2/latent_free/acdc_la/scripts/common.sh
run common.spatial_curve --n-clips ${N:-12000} --grids ${GRIDS:-16 8 4 2} --workers 4 --batch 48
echo "SPATIAL CURVE exit $? $(date)"
