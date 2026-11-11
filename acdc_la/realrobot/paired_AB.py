"""Is arm A (ours, latent) really worse than arm B (OpenVLA baseline) on the delta data?  (2026-11-11)

Both arms replay deterministically (`policy.reset(episode=ei)` seeds the flow sampler), so the 1.126 vs 1.054 deg
gap is exact, not sampling noise -- but it is still measured on only 8 held-out episodes, and steps inside an
episode are strongly correlated, so the step count (353) is NOT the sample size.  The independent unit is the
EPISODE.  This does a paired cluster bootstrap over the 8 episodes on the per-step dumps.

  envs/univla_vla/bin/python -m acdc_la.realrobot.paired_AB
"""
from math import comb

import numpy as np

S = "/tmp/claude-1690077540/-home-ti874297-Projects-latent-free/b5661022-4c10-49a3-82ca-0bd899324a1d/scratchpad"
a, b = np.load(f"{S}/steps_A.npz"), np.load(f"{S}/steps_B.npz")
assert (a["ep"] == b["ep"]).all(), "the two dumps must be the same steps in the same order"
ep = a["ep"]; eps = np.unique(ep)

def jmean(d):                                   # joints 0-5 only, mean over dims -> one number per step
    return d["first"][:, :6].mean(1), d["ens"][:, :6].mean(1)

A1, Ae = jmean(a); B1, Be = jmean(b)
rng = np.random.default_rng(0)

for lab, xa, xb in [("chunk[0]", A1, B1), ("ensembled", Ae, Be)]:
    per_ep = np.array([(xa[ep == e].mean(), xb[ep == e].mean()) for e in eps])
    d = per_ep[:, 0] - per_ep[:, 1]
    boot = np.array([(lambda s: (xa[np.isin(ep, eps[s])].mean() - xb[np.isin(ep, eps[s])].mean()))(
        rng.integers(0, len(eps), len(eps))) for _ in range(20000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n{lab}:  A {xa.mean():.3f} deg   B {xb.mean():.3f} deg   A-B {xa.mean()-xb.mean():+.3f} deg")
    print(f"  per-episode A-B: {np.round(d,3)}")
    k, n = int((d > 0).sum()), len(eps)
    p = min(2 * sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2**n, 1.0)
    print(f"  A worse in {k}/{n} episodes   (sign test two-sided p = {p:.3f})")
    print(f"  paired cluster bootstrap over {len(eps)} episodes, 20000 draws: 95% CI [{lo:+.3f}, {hi:+.3f}] deg")
    print(f"  -> {'SEPARATED (CI excludes 0)' if lo > 0 or hi < 0 else 'NOT separated (CI contains 0)'}")

print(f"\nwhole-chunk L1, normalised space:  A {a['norm_chunk'].mean():.4f}   B {b['norm_chunk'].mean():.4f}")
print("training-time oracle metric at step 10000: A 0.05634   B 0.06483   <- orders the arms the OTHER way")
