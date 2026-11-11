"""Rebuttal Table 4 (EAT ablation) from the train_multi validation dumps -- same protocols as the paper appendix.

    PYTHONPATH=. python -m acdc_la.downstream.eatab_table <arm> <seed>     # worker: one arm, cached
    PYTHONPATH=. python -m acdc_la.downstream.eatab_table                  # table from the cached rows

Columns (all NMSE = 1 - R^2, lower is better, clip-weighted over the sources with >= 80 test clips):
  Total = net (sum of the 4 segment displacements)   First = seg1   Tim. = prof (a_k - mean_k a_k)   Seq. = seq (12-d)
      -> nmse_bootstrap.per_source: per-source episode-disjoint ridge from frozen Z (APPENDIX_FACTS "Linear reader")
  Drop = 100 (1 - R^2_patch / R^2_clean) on net, moving-patch distractor, clean-fitted probe
  Act. = transported-action NMSE over all donors (appendix: EAT 0.724 clip-weighted, 0.722 median)
  Far  = transported-action NMSE for far donors (mean of donor-distance quartiles Q3, Q4)
  Near = the same for Q1, Q2;   Rise = Far - Near
      -> transport_by_source.per_source: frozen model-free reader DF -> action, applied to decode(F0, z_donor)
"""
import json, os, sys
import numpy as np

from ..paths import RESULTS
from . import nmse_bootstrap as NB
from . import transport_by_source as TB

ARMS = [("recon", "L_self"), ("a1", "L_self + L_inv"), ("noinv", "EAT w/o L_inv"), ("a2", "w/o L_effect"),
        ("a3", "w/o L_rel"), ("a4", "w/o L_dist"), ("a5", "L_effect -> head"), ("a6", "L_effect -> cycle"),
        ("full", "full EAT")]
CACHE = os.path.join(RESULTS, "eatab_rows")


def worker(arm, seed):
    os.makedirs(CACHE, exist_ok=True); tag = f"eatabm_{arm}_s{seed}"
    f = os.path.join(CACHE, f"{tag}.json")
    if os.path.exists(f): return
    probe = NB.per_source(tag)[0]
    d = np.load(os.path.join(RESULTS, f"{tag}_val.npz"), allow_pickle=True)
    tr = TB.per_source(tag, {"_DF": d["DF"].astype(np.float32)})
    json.dump(dict(probe=probe, transport=tr), open(f, "w"))
    print(f"done {tag}", flush=True)


def row(r):
    P, T = r["probe"], r["transport"]
    names = sorted(P); n = np.array([P[s]["n_test"] for s in names], float)
    out = NB.agg(P, names, n)
    ts = [s for s in T if all(f"Q{q}" in T[s] for q in (1, 2, 3, 4))]
    w = np.array([T[s]["n_test"] for s in ts], float)
    far = np.array([(T[s]["Q3"] + T[s]["Q4"]) / 2 for s in ts]); near = np.array([(T[s]["Q1"] + T[s]["Q2"]) / 2 for s in ts])
    al = np.array([T[s]["all"] for s in ts])
    out.update(act=float((al * w).sum() / w.sum()), far=float((far * w).sum() / w.sum()), near=float((near * w).sum() / w.sum()),
               far_median=float(np.median(far)), n_probe_src=len(names), n_transport_src=len(ts))
    out["rise"] = out["far"] - out["near"]
    return out


def main():
    if len(sys.argv) == 3: return worker(sys.argv[1], int(sys.argv[2]))
    table = {}
    for arm, lab in ARMS:
        seeds = {}
        for s in (0, 1, 2):
            f = os.path.join(CACHE, f"eatabm_{arm}_s{s}.json")
            if os.path.exists(f): seeds[s] = row(json.load(open(f)))
        if seeds: table[arm] = dict(label=lab, seeds=seeds)
    json.dump(table, open(os.path.join(RESULTS, "eatab_table.json"), "w"), indent=1)
    K = [("act", "Act."), ("net", "Total"), ("seg1", "First"), ("prof", "Tim."), ("seq", "Seq."), ("drop", "Drop%"), ("far", "Far"),
         ("near", "Near"), ("rise", "Rise")]
    print(f"{'row':22s}{'seeds':>6s}" + "".join(f"{h:>14s}" for _, h in K))
    for arm, t in table.items():
        v = {k: np.array([x[k] for x in t["seeds"].values()]) for k, _ in K}
        cell = lambda a: f"{a.mean():.3f}" + (f"±{a.std(ddof=1):.3f}" if len(a) > 1 else "      ")
        print(f"{t['label']:22s}{len(t['seeds']):>6d}" + "".join(f"{cell(v[k]):>14s}" if k != "drop" else
              f"{v[k].mean():>8.1f}" + (f"±{v[k].std(ddof=1):.1f}" if len(v[k]) > 1 else "      ") for k, _ in K))


if __name__ == "__main__":
    main()
