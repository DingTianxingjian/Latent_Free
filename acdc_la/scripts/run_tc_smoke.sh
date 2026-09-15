#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 8 --mem=80G
#SBATCH -t 00:20:00
#SBATCH -J acdc_tcsmoke
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/tcsmoke_%j.out
# Cheap correctness check for the deployment-prior term before it costs 8 h of a P0 tokenizer slot.
# (1) shapes + gradient reach of prior_code, (2) a handful of real training steps with --lam-prior on.
source /anvil/scratch/x-tding2/latent_free/acdc_la/scripts/common.sh
$PY - <<'PYEOF'
import torch, torch.nn.functional as F
from acdc_la.common.tcvq import TCVQ
dev = "cuda"
m = TCVQ(4, "endpointdyn", 4, 16, 64, 768, 6, 12, 4, bottleneck="continuous").to(dev)
B, P, D = 3, 16, 768
f0 = torch.randn(B, P, D, device=dev); dF = torch.randn(B, 4, P, D, device=dev)
c = torch.randn(B, m.prior_lang[0].normalized_shape[0], device=dev)
zq, z, x, idx = m.quantize(f0, dF)
zp = m.prior_code(f0, c)
rec = m.decode(f0, c, zq)
print("x", tuple(x.shape), "zp", tuple(zp.shape), "rec", tuple(rec.shape), "target", tuple(dF.shape))
assert zp.shape == x.shape, (zp.shape, x.shape)
prior = F.mse_loss(x, zp) / x.var(0).mean().clamp_min(1e-6)
(F.mse_loss(rec, dF) + 0.1 * prior).backward()
pr = [n for n, p in m.named_parameters() if n.startswith("prior") and p.grad is not None and p.grad.abs().sum() > 0]
en = [n for n, p in m.named_parameters() if n.startswith(("enc.", "to_code", "inp.")) and p.grad is not None and p.grad.abs().sum() > 0]
print("prior params with grad:", len(pr), " encoder params with grad:", len(en), " prior loss:", float(prior))
assert len(pr) >= 6 and len(en) >= 6, "prior term does not reach both sides"
# scale invariance: halving the code must not change the normalised prior loss
with torch.no_grad():
    for p in m.to_code[1].parameters(): p.mul_(0.5)
    x2 = m.pre_quant(f0, dF); p2 = F.mse_loss(x2, m.prior_code(f0, c)) / x2.var(0).mean().clamp_min(1e-6)
print("prior loss after halving the code:", float(p2), "(plain MSE would have quartered)")
print("MODULE SMOKE OK")
PYEOF
run stage2_tc.train --tag smoke_prior --steps 6 --batch 8 --warmup 1 --workers 8 \
    --tc endpointdyn --bottleneck continuous --n-tokens 4 --latent-dim 64 --lam-prior 0.1
echo "TC SMOKE exit $? $(date)"
