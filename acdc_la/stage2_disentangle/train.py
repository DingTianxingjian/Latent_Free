"""Stage II -- DISENTANGLE.  Split the AC/DC code into a task-centric channel z (per cell) and a
task-irrelevant channel u, defined by an INTERVENTION rather than by two heads that the loss treats
identically (UniVLA's TC/TI split, whose released weights do not separate: audit 2026-08).

Every training clip is rendered twice: clean, and with a moving natural-video patch whose motion is
independent of the robot (data.distractor).  With z_c, u_c and z_p, u_p the codes of the two renderings,

    L = L_rec(clean) + L_rec(patch) + beta KLs                                      (Stage I on both)
      + lam_inv  * || z_p - z_c ||^2                                                z must not see the nuisance
      + lam_xrec * || D(f0_p, c, z_c, u_p) - dF_p ||^2  + || D(f0_c, c, z_p, u_c) - dF_c ||^2
                                                                                    the clean code + the patch residual must
                                                                                    reproduce the patch clip: nuisance lives in u
Nothing here names the robot or reads an action label.  The metric is G_sep (stage2_disentangle/probes.py):
action R2 must be in z and not in u; patch-motion R2 must be in u and not in z.

    python -m acdc_la.stage2_disentangle.train --tag s2_tcti --init-from s1_acdc
`--lam-inv 0 --lam-xrec 0` is the Stage-I model trained on both renderings (the augmentation control).
"""
import argparse
import torch.nn.functional as F

from ..common.train_loop import common_args, run
from ..common.model import kl, reparam


def step(model, feat, b, a, ctx, train=True):
    zc_mu, zc_lv, uc_mu, uc_lv = model.encode(b["cells"]); zp_mu, zp_lv, up_mu, up_lv = model.encode(b["cells_patch"])
    zc, uc, zp, up = reparam(zc_mu, zc_lv, train), reparam(uc_mu, uc_lv, train), reparam(zp_mu, zp_lv, train), reparam(up_mu, up_lv, train)
    out = {}
    out["rec"] = F.mse_loss(model.decode(b["f0"], b["c"], zc, uc), b["dF"])
    out["rec_p"] = F.mse_loss(model.decode(b["f0_patch"], b["c"], zp, up), b["dF_patch"])
    out["kl_z"], out["kl_u"] = (kl(zc_mu, zc_lv) + kl(zp_mu, zp_lv)) / 2, (kl(uc_mu, uc_lv) + kl(up_mu, up_lv)) / 2
    out["inv"] = F.mse_loss(zp_mu, zc_mu)
    out["xrec"] = (F.mse_loss(model.decode(b["f0_patch"], b["c"], zc, up), b["dF_patch"]) +
                   F.mse_loss(model.decode(b["f0"], b["c"], zp, uc), b["dF"])) / 2
    loss = out["rec"] + out["rec_p"] + a.beta_z * out["kl_z"] + a.beta_u * out["kl_u"] + a.lam_inv * out["inv"] + a.lam_xrec * out["xrec"]
    out["loss"] = loss
    return loss, out


def main():
    ap = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--lam-inv", type=float, default=1.0); ap.add_argument("--lam-xrec", type=float, default=1.0)
    a = ap.parse_args(); a.needs_patch = True
    run(a, step, ramp_keys=("lam_inv", "lam_xrec"), stage="II")


if __name__ == "__main__":
    main()
