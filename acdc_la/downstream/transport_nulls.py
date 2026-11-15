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

    PYTHONPATH=. envs/univla_vla/bin/python -m acdc_la.downstream.transport_nulls [--full <tag>] [--residual <tag>]

--residual checks the RESIDUAL metric of `phys_replay.py:280`, cos(Psi(Y_hat - Y_donor), Psi(Y_real - Y_donor)),
proposed to answer "does the decoder use the recipient state, or just copy the donor".  copy-donor is 0 there by
construction, but cosine is SCALE-INVARIANT, so every predictor of the form `a * Y_donor` with a < 1 -- which uses
no recipient information at all -- gets the SAME positive score as predicting zero.  Only a = 1 is excluded, a
measure-zero point.  Fix: project the donor direction out of both sides first, which sends the whole family to 0
exactly; and compare a matched donor against a deranged one, or the remaining signal may be the first frame alone.
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


def residual(tag):
    """The phys_replay residual metric and its donor-orthogonalised repair, with a wrong-donor control."""
    from .transport_by_source import donor_index
    d = np.load(os.path.join(RESULTS, f"{tag}_val.npz"), allow_pickle=True)
    Yr = d["DF"].astype(np.float32); n = len(Yr); par = np.arange(n) ^ 1
    cos = lambda A, B: (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-9)
    Yd, Yh = Yr[par], d["DMATCH"].astype(np.float32)
    print(f"\n[{tag}] AS PROPOSED  cos(pred - donor, real - donor):")
    print(f"    copy_donor  (Yh = Yd)            {0.0:+.4f}  0 by construction")
    print(f"    ZERO        (Yh = 0)             {cos(-Yd, Yr - Yd).mean():+.4f}  <-- NO recipient information")
    for a in (0.25, 0.5, 0.75, 1.25):
        print(f"    scaled copy a={a:<5}              {cos((a - 1) * Yd, Yr - Yd).mean():+.4f}  <-- NO recipient information")
    print(f"    model       (Yh = DMATCH)        {cos(Yh - Yd, Yr - Yd).mean():+.4f}")
    print(f"    cos(real, donor) = {cos(Yr, Yd).mean():+.4f}: near-orthogonal, which is where the positive null comes from")

    def ortho(P_hat, P_don, keep):
        u = P_don / (np.linalg.norm(P_don, axis=1, keepdims=True) + 1e-9)
        pr = lambda A: A - (A * u).sum(1, keepdims=True) * u
        return float(cos(pr(P_hat), pr(Yr))[keep].mean())

    di = donor_index(d["ACT"], d["ASW"], d["GE"], d["DONOR_GE"], d["SI"]); ok = di >= 0
    print(f"[{tag}] REPAIRED  cos(P(pred), P(real)), P removing the donor direction:")
    print(f"    zero / any scaled copy           {0.0:+.4f}  0 for the WHOLE family, not one point")
    print(f"    model, matched donor             {ortho(Yh, Yd, np.ones(n, bool)):+.4f}")
    print(f"    model, WRONG (deranged) donor    {ortho(d['DSW'].astype(np.float32), Yr[np.where(ok, di, 0)], ok):+.4f}"
          f"   n={ok.sum()}  <-- the first-frame-only level")
    print("    the donor-specific claim is the GAP between the last two, not the distance from 0.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default=None, help="also compute the copy-donor reference for this tag (loads DF)")
    ap.add_argument("--residual", default=None, help="check the phys_replay residual metric on this tag")
    A = ap.parse_args()
    table()
    if A.full:
        full(A.full)
    if A.residual:
        residual(A.residual)
