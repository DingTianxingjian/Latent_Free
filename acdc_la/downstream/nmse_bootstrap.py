"""NMSE ablation table with bootstrap intervals, from existing validation dumps -- no training.

    PYTHONPATH=. ../envs/univla_vla/bin/python -m acdc_la.downstream.nmse_bootstrap

For every arm: the repo's own per-source, episode-disjoint ridge probe (common.probes.per_source_readouts, the
protocol behind every R^2 in the ledger) from the frozen latent Z to four action targets built from the clip's
action labels (K segments x xyz):  net = sum_k a_k,  seg1 = a_0,  seq = a (K*3),  prof = a - mean_k a.
NMSE = 1 - R^2 on each source's held-out episodes (targets are fitted per source, so this is the normalised
MSE of that source's linear action probe), aggregated clip-weighted over sources exactly as the ledger does.
The distractor drop reads the same clean-fitted probe on the moving-patch Z.

Intervals: 2,000 bootstrap resamples of the 29 SOURCES (the unit the aggregate is over), with the SAME resample
applied to every arm, so the difference of two arms gets a paired interval.  This measures how much the
reported number depends on which sources are in the mixture; it does not replace training seeds.
"""
import json
import os

import numpy as np

from ..common.probes import per_source_readouts
from ..paths import RESULTS

ARMS = [                        # label, dump tag
    ("endpoint only (N)",          "s1_mean_1ep"),
    ("DC only (N, explicit)",      "s1_dc"),
    ("N + A1, earlier tokenizer",  "s1_acdc_1ep"),
    ("TERRA start (65k)",          "tc_cont"),
    ("+15k reconstruction only",   "tc_cont_plus15k"),
    ("+15k EAT w/o invariance",    "latp2_anchor"),
    ("+15k EAT (TERRA)",           "latp2_inv"),
    ("UniVLA matched",             "univla_matched"),
    ("UniVLA, random TI codebook", "univla_noinh"),
]
TARGETS = ("net", "seg1", "prof", "seq")
B = 2000


def targets(act):
    a = np.asarray(act, np.float64)                                   # (n, K, 3)
    return dict(net=a.sum(1), seg1=a[:, 0], seq=a.reshape(len(a), -1),
                prof=(a - a.mean(1, keepdims=True)).reshape(len(a), -1))


def per_source(tag):
    d = np.load(os.path.join(RESULTS, f"{tag}_val.npz"), allow_pickle=True)
    P = per_source_readouts(d["Z"], targets(d["ACT"]), d["GE"], d["SI"], d["names"],
                            extra={"net_patch": (d["ZP"], "net", None)})
    rows = {s: r for s, r in P["per_source"].items() if r.get("n_test", 0) >= 80 and "r2_net" in r}
    return rows, {k: P.get(f"mean_r2_{k}") for k in TARGETS + ("net_patch",)}


def agg(rows, names, w):
    out = {}
    for k in TARGETS:
        v = np.array([rows[s][f"r2_{k}"] for s in names]); out[k] = 1 - float((v * w).sum() / w.sum())
    rc = np.array([rows[s]["r2_net"] for s in names]); rp = np.array([rows[s].get("r2_net_patch", np.nan) for s in names])
    out["drop"] = 100 * (1 - float((rp * w).sum() / w.sum()) / float((rc * w).sum() / w.sum()))
    return out


CACHE = os.path.join(RESULTS, "nmse_rows")


def probe_one(tag):
    """Per-source rows for one arm, cached so arms can run as parallel processes."""
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"{tag}.json")
    if not os.path.exists(f):
        json.dump(per_source(tag)[0], open(f, "w"))
    return json.load(open(f))


def main():
    import sys
    if len(sys.argv) > 1:                                  # worker: probe one arm and exit
        probe_one(sys.argv[1]); print(f"probed {sys.argv[1]}", flush=True); return
    per = {}
    for lab, tag in ARMS:
        if not os.path.exists(os.path.join(CACHE, f"{tag}.json")):
            print(f"  (missing {tag})"); continue
        per[lab] = probe_one(tag)
    common = sorted(set.intersection(*[set(r) for r in per.values()]))
    n = np.array([per[next(iter(per))][s]["n_test"] for s in common], float)
    point = {lab: agg(r, common, n) for lab, r in per.items()}
    rs = np.random.RandomState(0); boot = {lab: {k: [] for k in TARGETS + ("drop",)} for lab in per}
    for _ in range(B):
        idx = rs.randint(0, len(common), len(common)); names = [common[i] for i in idx]; w = n[idx]
        for lab, r in per.items():
            for k, v in agg(r, names, w).items(): boot[lab][k].append(v)
    ci = {lab: {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in b.items()}
          for lab, b in boot.items()}
    ref = "+15k EAT (TERRA)"
    paired = {}
    if ref in boot:
        for lab in boot:
            if lab == ref: continue
            paired[lab] = {k: [float(np.percentile(np.array(boot[ref][k]) - np.array(boot[lab][k]), q)) for q in (2.5, 97.5)]
                           for k in TARGETS + ("drop",)}
    out = dict(n_sources=len(common), bootstrap=B, point=point, ci95=ci, paired_ci95_terra_minus_arm=paired)
    json.dump(out, open(os.path.join(RESULTS, "nmse_bootstrap.json"), "w"), indent=1)
    print(f"\nNMSE = 1 - R^2 (lower is better); 95% source-bootstrap interval; {len(common)} sources, {B} resamples")
    print(f"{'arm':30s}" + "".join(f"{k:>18s}" for k in TARGETS) + f"{'distractor drop %':>22s}")
    for lab in per:
        cells = "".join(f"  {point[lab][k]:.3f} [{ci[lab][k][0]:.3f},{ci[lab][k][1]:.3f}]" for k in TARGETS)
        print(f"{lab:30s}{cells}    {point[lab]['drop']:5.1f} [{ci[lab]['drop'][0]:.1f},{ci[lab]['drop'][1]:.1f}]")
    if paired:
        print(f"\npaired 95% interval of (TERRA - arm): negative NMSE / negative drop = TERRA better")
        for lab, p in paired.items():
            print(f"{lab:30s}" + "".join(f"  [{p[k][0]:+.3f},{p[k][1]:+.3f}]" for k in TARGETS)
                  + f"    [{p['drop'][0]:+.1f},{p['drop'][1]:+.1f}]")


if __name__ == "__main__":
    main()
