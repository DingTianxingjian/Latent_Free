#!/bin/bash
# Build the three venvs + third-party checkouts on UCF Newton (see REPRODUCE.md §1).
# Layout: $BASE/latent_free (this repo), $BASE/envs/{latent_free,libero,univla_vla,LIBERO}.
# Safe to re-run: existing checkouts/venvs are reused.  Usage: bash acdc_la/envs/setup_newton.sh [latent_free|libero|univla_vla ...]
set -euo pipefail
BASE=/lustre/fs1/home/ti874297/lf
LF=$BASE/latent_free
ENVS=$BASE/envs
UV=${UV:-$HOME/.local/bin/uv}
CU121=https://download.pytorch.org/whl/cu121
# Newton login nodes cap threads per user; keep uv/rayon/git small.
export UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-4} UV_CONCURRENT_INSTALLS=${UV_CONCURRENT_INSTALLS:-2} \
       UV_CONCURRENT_BUILDS=1 RAYON_NUM_THREADS=2 TOKIO_WORKER_THREADS=2 UV_LINK_MODE=copy \
       UV_CACHE_DIR=$BASE/uv_cache UV_PYTHON_INSTALL_DIR=$BASE/uv_python UV_HTTP_TIMEOUT=300
TARGETS=${*:-latent_free libero univla_vla}
TMP=$(mktemp -d); trap 'rm -rf $TMP' EXIT
# frozen lists carry Anvil -e paths; point them at Newton
fix_req() { sed "s|/anvil/scratch/x-tding2|$BASE|g" "$LF/acdc_la/envs/$1.requirements.txt" > "$TMP/$1.txt"; echo "$TMP/$1.txt"; }

cd "$LF"
if [ ! -d third_party/UniVLA/.git ]; then
  git -c pack.threads=1 clone https://github.com/OpenDriveLab/UniVLA.git third_party/UniVLA
  git -C third_party/UniVLA checkout -q 0ab9e9dd3074ffa14fdd1a00aa2aa84e4f8e1352
  (cd third_party/UniVLA && git apply ../../acdc_la/patches/*.patch)
fi
[ -d "$ENVS/LIBERO/.git" ] || git -c pack.threads=1 clone https://github.com/Lifelong-Robot-Learning/LIBERO "$ENVS/LIBERO"

for t in $TARGETS; do
  echo "=== $t"
  [ -x "$ENVS/$t/bin/python" ] || "$UV" venv "$ENVS/$t" --python 3.10
  export VIRTUAL_ENV=$ENVS/$t
  case $t in
    latent_free)
      "$UV" pip install --index-strategy unsafe-best-match --extra-index-url $CU121 \
        torch==2.4.1 torchvision==0.19.1 numpy scipy h5py timm tqdm pillow -r "$(fix_req latent_free)" ;;
    libero)
      "$UV" pip install --index-strategy unsafe-best-match --extra-index-url $CU121 -r "$(fix_req libero)" ;;
    univla_vla)
      # the freeze (what actually ran) overrides univla's own pins (e.g. absl-py), so install univla without deps
      req=$(fix_req univla_vla); grep -v "third_party/UniVLA" "$req" > "$TMP/univla_nodeps.txt"
      # the freeze is not self-consistent (e.g. protobuf 3.20.3 next to googleapis-common-protos needing >=6): install it verbatim
      "$UV" pip install --no-deps --index-strategy unsafe-best-match --extra-index-url $CU121 -r "$TMP/univla_nodeps.txt"
      "$UV" pip install --no-deps -e "$LF/third_party/UniVLA" ;;
  esac
  unset VIRTUAL_ENV
done
echo "=== setup_newton done"
