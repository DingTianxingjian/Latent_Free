#!/bin/bash
#SBATCH --account=ytian
#SBATCH --partition=normal
#SBATCH --exclude=evc22,evc26,evc38,evc40,evc43,evc44,evc45,evc48
#SBATCH -N 1 -n 1 -c 4 --mem=16G
#SBATCH -t 12:00:00
#SBATCH -J hf_terra
#SBATCH -o /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/hf_terra_%j.out
# Push post-trained policies to the private HF model repo Potestates/TERRA (2026-10-29), one subdirectory per policy,
# over git + git-lfs with the SSH key ~/.ssh/id_ed25519_hf (no API token on Newton).  Optimiser state is left out.
#   SUB=libero_10_full RUN=<run dir> sbatch acdc_la/scripts/hf_upload_terra.sh
set -eu -o pipefail
SUB=${SUB:?}; RUN=${RUN:?}
W=/lustre/fs1/home/ti874297/lf/hf_upload/TERRA
if [ ! -d $W/.git ]; then mkdir -p $(dirname $W); GIT_LFS_SKIP_SMUDGE=1 git clone git@hf.co:Potestates/TERRA $W; fi
cd $W && git lfs install --local && git pull --ff-only
cat > .gitattributes <<'EOF'
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
tokenizer.json filter=lfs diff=lfs merge=lfs -text
EOF
[ -f README.md ] && grep -q "TERRA" README.md || cat > README.md <<'EOF'
---
private: true
---
# TERRA — post-trained LIBERO policies

OpenVLA-7B with TERRA latent-action phase 2 (`latp2_inv` labels), post-trained on LIBERO (flow head, 4 query tokens).
One directory per policy: merged weights (`model-*.safetensors`), action/latent heads (`action_decoder.pt`),
`dataset_statistics.json` (action un-normalisation), `ctrl_meta.json`, processor / tokenizer.
Evaluate with `acdc_la.downstream.vla.run_libero_eval_cont --pretrained_checkpoint <dir>` (repo Latent_Free, branch newton).

| dir | suite | demos | standard LIBERO (500 ep.) |
|---|---|---|---|
EOF
mkdir -p $SUB
rsync -a --exclude trainer_state.pt "$RUN"/ $SUB/
grep -q "^| $SUB " README.md || echo "| $SUB | $(python3 -c "import json;print(json.load(open('$SUB/ctrl_meta.json'))['dataset_name'])") | ${DEMOS:-full} | ${SCORE:--} |" >> README.md
git add .gitattributes README.md $SUB
git -c user.name=DingTianxingjian -c user.email=edmund330109@gmail.com commit -q -m "Add $SUB" || git diff --cached --quiet
git push origin main
echo "HF upload $SUB done $(date)"; du -sh $SUB
