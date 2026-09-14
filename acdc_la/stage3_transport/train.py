"""Stage III -- TRANSPORT.  Latent Action Transport Pretraining (LATP).

The tokenizer is FROZEN (grad_E L = 0): z means what Stage I made it mean.  Only the state-conditioned
transport operator T(s, z; u) = decode(f0, c, z, u) is trained, and its off-diagonal supervision needs
no action labels and no pairs:

    L = L_rec + beta KLs                                     (diagonal: own code on own state)
      + lam_real * MMD( {T(s_j, z_{pi(j)})}_j , {dF_j}_j )    swapped transitions must be distributed like real ones
      + lam_cyc  * || E(T(s_j, z_i)) - z_i ||^2               and must still carry the donor's code
with pi a same-source, cross-episode derangement.  Ledger evidence (2026-09-08, controlled tokenizer
rec_src): decoder-only training lifts the independent transfer cosine 0.449 -> 0.512 at zero clean cost,
above all three full-training swap seeds (0.485-0.492) -- the gain lives in the operator, not in z.

    python -m acdc_la.stage3_transport.train --tag s3_latp --init-from s1_acdc             # frozen encoder = the method
    python -m acdc_la.stage3_transport.train --tag s3_full --init-from s1_acdc --train-encoder   # ablation: full swap
    python -m acdc_la.stage3_transport.train --tag s3_ctrl --init-from s1_acdc --lam-real 0 --lam-cyc 0   # decoder-only rec (control)
Legacy equivalents: swap_fz (method), swap_src (full), rec_fz (control).
"""
import argparse
import torch
import torch.nn.functional as F

from ..common.train_loop import common_args, run
from ..common.model import kl, reparam, mmd, derangement


def step(model, feat, b, a, ctx, train=True):
    zmu, zlv, umu, ulv = model.encode(b["cells"])
    z, u = reparam(zmu, zlv, train), reparam(umu, ulv, train)
    out = {}
    rec = model.decode(b["f0"], b["c"], z, u)
    out["rec"], out["kl_z"], out["kl_u"] = F.mse_loss(rec, b["dF"]), kl(zmu, zlv), kl(umu, ulv)
    loss = out["rec"] + a.beta_z * out["kl_z"] + a.beta_u * out["kl_u"]
    B = z.shape[0]; pi = derangement(B, ctx["dev"], b["ge"])
    sw = model.decode(b["f0"], b["c"], z[pi], u)                          # donor i's effect realised on recipient j's state
    out["mmd"] = mmd(sw, b["dF"], ctx["proj"])
    out["cyc"] = F.mse_loss(model.encode(feat.cells_of(sw))[0], zmu[pi].detach())
    loss = loss + a.lam_real * out["mmd"] + a.lam_cyc * out["cyc"]
    with torch.no_grad():
        pk = derangement(B, ctx["dev"], b["ge"])
        out["swap_sens"] = ((sw - model.decode(b["f0"], b["c"], z[pk], u)).flatten(1).norm(dim=1) / b["dF"].flatten(1).norm(dim=1).clamp_min(1e-3)).mean()
    out["loss"] = loss
    return loss, out


def main():
    ap = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--lam-real", type=float, default=1.0); ap.add_argument("--lam-cyc", type=float, default=1.0)
    ap.add_argument("--train-encoder", action="store_true", help="ablation: let the swap objective update the tokenizer (full swap)")
    a = ap.parse_args()
    assert a.init_from or a.eval_only, "Stage III starts from a Stage-I/II tokenizer (--init-from)"
    a.freeze_encoder = not a.train_encoder
    run(a, step, ramp_keys=("lam_real", "lam_cyc"), stage="III")


if __name__ == "__main__":
    main()
