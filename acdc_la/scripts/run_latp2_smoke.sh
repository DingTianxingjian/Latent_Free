#!/bin/bash
#SBATCH --account=cis250950-ai
#SBATCH --partition=ai
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 8 --mem=80G
#SBATCH -t 00:20:00
#SBATCH -J acdc_latp2sm
#SBATCH -o /anvil/scratch/x-tding2/latent_free/acdc_la/logs/latp2smoke_%j.out
# Plumbing check for LATP v2 with a synthetic batch: no checkpoint, no data, no loader.
# Asserts the three things that would silently ruin the run: the EMA target actually lags, L_recover reaches
# the ONLINE ENCODER (the whole point of v2), and RTC reaches the decoder but NOT the encoder.
source /anvil/scratch/x-tding2/latent_free/acdc_la/scripts/common.sh
$PY - <<'PYEOF'
import argparse, torch, copy
from acdc_la.common.tcvq import TCVQ
from acdc_la.common.model import jl_projection
from acdc_la.stage3_transport import train_v2 as V

dev = "cuda"; K, P, D, B = 4, 16, 768, 12
m = TCVQ(K, "endpointdyn", 4, 16, 64, 768, 6, 12, 4, bottleneck="continuous").to(dev)
a = argparse.Namespace(lam_recover=1.0, lam_real=1.0, lam_rel=0.5, recover_l1=1.0, tau=0.9, buffer=4, recover_detach=False,
                       rtc_groups=4, psi_dim=512, probe_noise=0.1, lam_var=0.0)
ctx = dict(dev=dev, proj=jl_projection(K * P * D, 512, 0, dev))
def batch(seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return dict(f0=torch.randn(B, P, D, device=dev), dF=torch.randn(B, K, P, D, device=dev),
                c=torch.randn(B, m.prior_lang[0].normalized_shape[0], device=dev),
                si=torch.randint(0, 3, (B,), generator=g), ge=torch.arange(B) + 100 * seed)
# The channel test: PURE RANDOM tensors carry no action semantics, so anything L_recover achieves here is a channel
# E and T built together.  Run both routings and compare rec_cos (can be a channel) with rec_frozen (cannot: the
# frozen Phase-1 operator never received the recovery gradient).
for det in (False, True):
    mm = copy.deepcopy(m); cc = dict(dev=dev, proj=ctx["proj"]); aa = copy.deepcopy(a); aa.recover_detach = det
    oo = torch.optim.AdamW([p for p in mm.parameters() if p.requires_grad], lr=1e-4)
    for s in range(12):
        l, o = V.step(mm, None, batch(s), aa, cc); oo.zero_grad(); l.backward(); oo.step()
    print(f"[channel] recover_detach={det}: rec_cos={float(o['rec_cos']):.4f} rec_shuffle={float(o['rec_shuffle']):.4f} "
          f"MARGIN={float(o['rec_margin']):.4f} rec_frozen={float(o['rec_frozen']):.4f} "
          f"z_erank={float(o['z_erank']):.2f}/{m.d} zt_erank={float(o['zt_erank']):.2f} "
          f"-- margin<=0 means COLLAPSE, not recovery")

opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
for s in range(6):
    loss, out = V.step(m, None, batch(s), a, ctx)
    opt.zero_grad(); loss.backward()
    if s == 4:
        enc = sum(p.grad.abs().sum().item() for n, p in m.named_parameters()
                  if n.startswith(("enc.", "to_code", "inp.", "query", "pos_emb", "type_emb")) and p.grad is not None)
        dec = sum(p.grad.abs().sum().item() for n, p in m.named_parameters() if n.startswith("dec.") and p.grad is not None)
        assert enc > 0 and dec > 0, (enc, dec)
        print(f"grad reaching online encoder {enc:.3e}  decoder {dec:.3e}")
    opt.step()
    print(f"step {s} " + " ".join(f"{k}={float(v):.4f}" for k, v in out.items()))
lag = max((pd - ps).abs().max().item() for pd, ps in zip(ctx["teacher"].parameters(), m.parameters()))
assert lag > 0, "EMA target has not separated from the online encoder"
print(f"teacher-online max|diff| {lag:.3e}")
# RTC must reach the decoder only: with every other term off, the encoder must receive nothing
m.zero_grad(set_to_none=True)
a2 = copy.deepcopy(a); a2.lam_recover = 0.0; a2.lam_real = 0.0; a2.lam_rel = 1.0
loss, out = V.step(m, None, batch(9), a2, ctx); (loss - out["rec"]).backward()
enc = sum(p.grad.abs().sum().item() for n, p in m.named_parameters()
          if n.startswith(("enc.", "to_code", "inp.")) and p.grad is not None)
print(f"RTC-only gradient into the encoder: {enc:.3e} (must be 0)"); assert enc == 0.0, enc
print("LATP2 SMOKE OK")
PYEOF
echo "LATP2 SMOKE exit $? $(date)"
