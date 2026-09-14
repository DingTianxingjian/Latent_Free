"""Stage I -- PRESERVE.  Train the AC/DC tokenizer + decoder by reconstruction.

    L = || D(f0, c, z, u) - dF ||^2  +  beta_z KL(z)  +  beta_u KL(u)

The stage's claim is about the INPUT operator, not the loss: `--pool acdc --modes 2` keeps DC and
AC1 of every quadrant (8 cells); `--modes 1` keeps DC only; `--pool mean` / `--pool endpoint` are the
pooled tokenizers of the field (UniVLA's frame pair is `mean` at K=1).  All of them train with this
file and are compared by stage1_preserve/probes.py on the same validation dump.

    python -m acdc_la.stage1_preserve.train --tag s1_acdc            # DC+AC1 (the method)
    python -m acdc_la.stage1_preserve.train --tag s1_dc   --modes 1  # DC only
    python -m acdc_la.stage1_preserve.train --tag s1_mean --pool mean
Legacy equivalent: claude_survey/ckpt/swaplam_rec_src.pt (arm rec, T=5, stride 4, 10k steps, seed 0).
"""
import argparse
import torch.nn.functional as F

from ..common.train_loop import common_args, run
from ..common.model import kl, reparam


def step(model, feat, b, a, ctx, train=True):
    zmu, zlv, umu, ulv = model.encode(b["cells"])
    z, u = reparam(zmu, zlv, train), reparam(umu, ulv, train)
    rec = model.decode(b["f0"], b["c"], z, u)
    out = dict(rec=F.mse_loss(rec, b["dF"]), kl_z=kl(zmu, zlv), kl_u=kl(umu, ulv))
    loss = out["rec"] + a.beta_z * out["kl_z"] + a.beta_u * out["kl_u"]
    out["loss"] = loss
    return loss, out


def main():
    a = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)).parse_args()
    run(a, step, stage="I")


if __name__ == "__main__":
    main()
