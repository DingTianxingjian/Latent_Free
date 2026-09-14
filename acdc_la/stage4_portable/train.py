"""Stage IV -- PORTABLE CORE + RECONSTRUCTION RESIDUAL (review 2026-09-13).  See common/pcrr.py.

Starts from a Stage-III (LATP) checkpoint: the tokenizer stays frozen, the transport decoder is re-used and
two heads split its frozen code z0 into a portable core z^P (d_p) and a residual r (d_r):

    L = || D(s_i, z^P_i, r_i) - dF_i ||^2  +  beta_z KL(z^P)  +  beta_u KL(r)          self path, residual allowed
      + lam_real * MMD( {D(s_j, z^P_pi(j), 0)}_j , {dF_j}_j )                           cross path, residual forbidden
      + lam_cyc  * || P(E(D(s_j, z^P_i, 0))) - z^P_i ||^2                               ... and it must carry the donor's core
Logged diagnostics: self_nor = self reconstruction with r = 0 (what the residual buys), swap_sens.

    python -m acdc_la.stage4_portable.train --tag s4_pcrr --init-from s3_latp_1ep --beta-u 2e-4
Control for "10k more decoder steps": stage3_transport.train --tag s3_latp_1ep_cont --init-from s3_latp_1ep.
"""
import argparse
import torch
import torch.nn.functional as F

from ..common.train_loop import common_args, run, resolve_ckpt
from ..common.model import kl, reparam, mmd, derangement, load_ckpt
from ..common.pcrr import PCRR


def build(a, dev):
    base, dstats, _ = load_ckpt(resolve_ckpt(a.init_from), dev)
    assert not hasattr(base, "ckpt_meta"), "init from a LAM checkpoint (Stage I-III), not a Stage-IV one"
    assert base.K == a.T - 1 and base.n_cells == (4 * a.modes if a.pool == "acdc" else 4), (base.K, base.n_cells)
    return PCRR(base, a.d_p, a.d_r).to(dev), dstats


def step(model, feat, b, a, ctx, train=True):
    pmu, plv, rmu, rlv = model.encode(b["cells"])
    zp, r = reparam(pmu, plv, train), reparam(rmu, rlv, train)
    out = {}
    rec = model.decode(b["f0"], b["c"], zp, r)
    out["rec"], out["kl_z"], out["kl_u"] = F.mse_loss(rec, b["dF"]), kl(pmu.flatten(1), plv.flatten(1)), kl(rmu, rlv)
    loss = out["rec"] + a.beta_z * out["kl_z"] + a.beta_u * out["kl_u"]
    B = zp.shape[0]; r0 = torch.zeros_like(r); pi = derangement(B, ctx["dev"], b["ge"])
    sw = model.decode(b["f0"], b["c"], zp[pi], r0)                        # donor i's core realised on recipient j, no residual
    out["mmd"] = mmd(sw, b["dF"], ctx["proj"])
    out["cyc"] = F.mse_loss(model.encode(feat.cells_of(sw))[0], pmu[pi].detach())
    loss = loss + a.lam_real * out["mmd"] + a.lam_cyc * out["cyc"]
    with torch.no_grad():
        out["self_nor"] = F.mse_loss(model.decode(b["f0"], b["c"], zp, r0), b["dF"])
        pk = derangement(B, ctx["dev"], b["ge"])
        out["swap_sens"] = ((sw - model.decode(b["f0"], b["c"], zp[pk], r0)).flatten(1).norm(dim=1) / b["dF"].flatten(1).norm(dim=1).clamp_min(1e-3)).mean()
    out["loss"] = loss
    return loss, out


def main():
    ap = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--d-p", type=int, default=192); ap.add_argument("--d-r", type=int, default=64)
    ap.add_argument("--lam-real", type=float, default=1.0); ap.add_argument("--lam-cyc", type=float, default=1.0)
    a = ap.parse_args()
    assert a.init_from or a.eval_only, "Stage IV starts from a Stage-III checkpoint (--init-from)"
    run(a, step, ramp_keys=("lam_real", "lam_cyc"), stage="IV", build=build)


if __name__ == "__main__":
    main()
