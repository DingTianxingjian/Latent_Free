"""What is the RIGHT null for a transported prediction?  (2026-11-15)

Asked by the user: how do we show the latent does not merely MEMORISE / paste the donor's observed effect?
The registered simulator experiment (LEDGER 2026-10-27) scores `decode(f0_recipient, z_donor)` against
copy-donor / zero-effect references.  This checks what those references are actually worth, offline, on every
validation dump in acdc_la/results/ -- no checkpoint, no simulator, CPU only.

The val set is built as stratified NEAREST-ACTION PAIRS per source (partner of row i is i^1, a different episode),
so the dump already contains an approximate real counterfactual: the partner performed almost the same action in
another clip, and `DF[i^1]` is what that action really produced THERE.  Hence:

    zero        predict no change                      rel. MSE = 1.0 by construction
    copy        DF[i^1], the donor's own real effect    "just reuse what happened to the donor"
    E_CORRECT   decode(f0_i, z_partner)  (u zeroed)     the model's transported prediction
    E_WRONG     decode(f0_i, z_deranged) (u zeroed)     a wrong donor's code
    E_SELF      decode(f0_i, z_i)        (u zeroed)     upper bound, it saw its own code

All relative to the clip's OWN real future DF[i]: rel = mean((P - DF)^2) / mean(DF^2).

    PYTHONPATH=. envs/univla_vla/bin/python -m acdc_la.downstream.transport_nulls [--full <tag>]
"""
import argparse, glob, os
import numpy as np

from ..paths import RESULTS


def table():
    print(f"{'arm':34s} {'n':>6s} {'E_SELF':>8s} {'E_CORR':>8s} {'E_WRONG':>8s} {'disc%':>7s}  vs do-nothing")
    print("-" * 94)
    for p in sorted(glob.glob(os.path.join(RESULTS, "*_val.npz"))):
        tag = os.path.basename(p)[:-8]
        d = np.load(p, allow_pickle=True)
        if "E_CORRECT" not in d.files:
            continue
        es, ec, ew = (float(d[k].mean()) for k in ("E_SELF", "E_CORRECT", "E_WRONG"))
        disc = 100 * float((d["E_CORRECT"] < d["E_WRONG"]).mean())
        print(f"{tag:34s} {len(d['E_CORRECT']):6d} {es:8.3f} {ec:8.3f} {ew:8.3f} {disc:7.1f}"
              f"  {'BEATS' if ec < 1.0 else 'worse than'} predicting no change")


def full(tag):
    """The copy-donor reference needs DF, which is 1.4 GB per dump -- so only on request."""
    d = np.load(os.path.join(RESULTS, f"{tag}_val.npz"), allow_pickle=True)
    DF = d["DF"].astype(np.float32); n = len(DF); par = np.arange(n) ^ 1
    nrm = (DF ** 2).mean(1)
    rel = lambda P: float((((P - DF) ** 2).mean(1) / nrm).mean())
    print(f"\n[{tag}] partner is a different episode in {100*(d['GE'][par]!=d['GE']).mean():.0f} % of rows, "
          f"a different source in {100*(d['SI'][par]!=d['SI']).mean():.0f} %")
    a = d["ACT"].reshape(n, -1)
    rnd = np.random.default_rng(0).permutation(n)
    print(f"[{tag}] action distance: partner {np.linalg.norm(a-a[par],axis=1).mean():.3f}  "
          f"random {np.linalg.norm(a-a[rnd],axis=1).mean():.3f}")
    print(f"[{tag}] relative MSE against the clip's own real future (lower better):")
    print(f"    zero / do-nothing                        {rel(np.zeros_like(DF)):.4f}")
    print(f"    COPY the donor's real effect DF[i^1]     {rel(DF[par]):.4f}")
    print(f"    model, matched donor   (E_CORRECT)       {float(d['E_CORRECT'].mean()):.4f}")
    print(f"    model, wrong   donor   (E_WRONG)         {float(d['E_WRONG'].mean()):.4f}")
    print(f"    model, own code        (E_SELF)          {float(d['E_SELF'].mean()):.4f}")
    copy = ((DF[par] - DF) ** 2).mean(1) / nrm
    print(f"    the model beats COPY on {100*(d['E_CORRECT']<copy).mean():.1f} % of clips -- "
          f"which is why COPY is a straw man, not a control")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default=None, help="also compute the copy-donor reference for this tag (loads DF)")
    A = ap.parse_args()
    table()
    if A.full:
        full(A.full)
