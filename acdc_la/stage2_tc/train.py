"""Stage-2 TC: a task-centric tokenizer trained with a NATIVE discrete bottleneck, at UniVLA's exact token budget.

    L = || D(f0, c, Q(E(cells))) - dF ||^2  +  lam_commit * || E(cells) - sg[Q] ||^2  +  || Q - sg[E(cells)] ||^2

with Q = UniVLA's VectorQuantizer (straight-through, dead-code restart every --restart-every steps).  See common/tcvq.py
for the ladder: --tc net (endpoint only, = their downstream interface), netdyn (the method), dyn.

    python -m acdc_la.stage2_tc.train --tag tc_netdyn --tc netdyn --steps 128000 --batch 64
    python -m acdc_la.stage2_tc.train --tag tc_net    --tc net    --steps 128000 --batch 64
"""
import argparse
import torch
import torch.nn.functional as F

from ..common.train_loop import common_args, run
from ..common.tcvq import TCVQ


def build(a, dev):
    return TCVQ(a.T - 1, a.tc, a.n_tokens, a.codebook, a.latent_dim, a.width, a.enc_layers,
                a.heads, a.layers).to(dev), None                                   # dstats: estimated by run()


def step(model, feat, b, a, ctx, train=True):
    z_q, z, x, idx = model.quantize(b["f0"], b["dF"])
    rec = model.decode(b["f0"], b["c"], z_q)
    out = dict(rec=F.mse_loss(rec, b["dF"]), codebook=F.mse_loss(z, x.detach()), commit=F.mse_loss(x, z.detach()))
    loss = out["rec"] + out["codebook"] + a.lam_commit * out["commit"]
    with torch.no_grad():
        out["codes"] = torch.tensor(float(len(torch.unique(idx))))                   # codebook usage in the batch
        out["perplex"] = torch.exp(-(lambda p: (p[p > 0] * p[p > 0].log()).sum())(torch.bincount(idx.flatten(), minlength=model.codebook_size).float() / idx.numel()))
    out["loss"] = loss
    return loss, out


def main():
    ap = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--tc", default="endpointdyn", choices=["endpoint", "endpointdyn", "dyn", "netonly"],
                    help="what the encoder sees: endpoint = [F_0, F_T-F_0] (UniVLA's evidence), endpointdyn = + AC1")
    ap.add_argument("--width", type=int, default=768); ap.add_argument("--enc-layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--n-tokens", type=int, default=4, help="policy tokens per clip (UniVLA: 4)")
    ap.add_argument("--codebook", type=int, default=16, help="codebook entries (UniVLA: 16)")
    ap.add_argument("--latent-dim", type=int, default=128, help="code dimension (UniVLA: 128)")
    ap.add_argument("--lam-commit", type=float, default=0.25)
    ap.add_argument("--restart-every", type=int, default=5000, help="dead-code restart period (UniVLA restarts per epoch)")
    a = ap.parse_args()
    run(a, step, stage="TC", build=build, on_step=lambda m, s: (m.vq.random_restart(), m.vq.reset_usage()) if a.restart_every and (s + 1) % a.restart_every == 0 else None)


if __name__ == "__main__":
    main()
