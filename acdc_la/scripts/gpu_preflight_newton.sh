# Sourced at the top of Newton GPU jobs.  Newton has nodes whose allocated GPUs do not work (nvidia-smi sees fewer
# devices than Slurm allocated, torch fails with "CUDA unknown error"; seen on evc22/38/40/43/44/45/48, 2026-10-12).
# If the allocation is not fully usable: record the node in acdc_la/logs/bad_nodes.txt and stop at once (exit 75),
# instead of hanging in the CUDA import for 20 minutes.  Resubmit by hand with the node added to --exclude.
_PF_PY=${PY:-/lustre/fs1/home/ti874297/lf/envs/univla_vla/bin/python}
_PF_WANT=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | grep -c .)
# 900 s: under heavy Lustre load the torch import alone has taken > 180 s, and a timeout was then misreported as a bad
# node (evc102/evc27, 2026-10-17, while evc102 was training fine).  A timeout is now reported as such, not as "usable 0".
_PF_T0=$(date +%s)
_PF_OUT=$(timeout 900 $_PF_PY -c "
import torch
n = torch.cuda.device_count()
for i in range(n): torch.zeros(8, device=f'cuda:{i}').sum().item()
print('PF_OK', n)" 2>&1); _PF_RC=$?
_PF_GOT=$(echo "$_PF_OUT" | awk '/^PF_OK/ {print $2}' | tail -1)
echo "[preflight] check took $(( $(date +%s) - _PF_T0 )) s, rc=$_PF_RC"
if [ $_PF_RC = 124 ]; then echo "[preflight] TIMEOUT (900 s) -- not a GPU verdict; stopping"; exit 75; fi
[ -z "$_PF_GOT" ] && echo "$_PF_OUT" | grep -iE "error|cuda" | tail -3 | sed 's/^/[preflight] /'
if [ "$_PF_WANT" = 0 ] || [ "${_PF_GOT:-0}" != "$_PF_WANT" ]; then
  echo "[preflight] BAD NODE $(hostname -s): Slurm allocated $_PF_WANT GPU(s), usable ${_PF_GOT:-0} -- stopping $(date)"
  echo "$(hostname -s) $(date +%F_%T) job=${SLURM_JOB_ID:-} alloc=$_PF_WANT usable=${_PF_GOT:-0}" >> /lustre/fs1/home/ti874297/lf/latent_free/acdc_la/logs/bad_nodes.txt
  exit 75
fi
echo "[preflight] $(hostname -s): $_PF_GOT/$_PF_WANT GPU(s) usable"
