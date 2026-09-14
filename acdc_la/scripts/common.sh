# sourced by every run script
set -u -o pipefail
cd /anvil/scratch/x-tding2/latent_free
PY=/anvil/scratch/x-tding2/envs/latent_free/bin/python
QUIET='xFormers|warnings.warn|Using cache|UserWarning|enable_nested'
run() {   # run <module> <args...>  -- python -m acdc_la.<module>, noise filtered, exit code preserved
  local mod=$1; shift
  $PY -u -m acdc_la.$mod "$@" 2>&1 | grep --line-buffered -vE "$QUIET"
  local rc=${PIPESTATUS[0]}; echo "[run] acdc_la.$mod $* -> exit $rc"; return $rc
}
