"""Transported-action NMSE per source and per donor-distance quartile, for reconstruction-only vs full EAT.

    PYTHONPATH=. ../envs/univla_vla/bin/python -m acdc_la.downstream.transport_by_source

From each arm's validation dump: DSW = decode(recipient F0, z of a same-source deranged donor), ASW = that donor's own
action.  A frozen reader is fitted per source on REAL transitions (DF -> 12-step action sequence, the repo's ridge,
train episodes only); DF is model-free, so both arms are read by the identical reader.  It is applied to the
transported DSW of the test episodes and compared with the donor's action: NMSE = 1 - R^2 (the repo's _r2).
Donor distance = || F0Q(recipient) - F0Q(donor) ||; the donor row is recovered exactly as the same-source row whose
episode is DONOR_GE and whose ACT equals ASW.  Quartiles are taken within each source over its test rows."""
import json
import os

import numpy as np

from ..common.probes import _r2, _ridge, standardise
from ..paths import RESULTS
from .nmse_bootstrap import targets

ARMS = {"recon": "tc_cont_plus15k", "eat": "latp2_inv"}


def donor_index(ACT, ASW, GE, DGE, SI):
    """Row of each clip's donor: same source, episode DONOR_GE, action identical to ASW."""
    n = len(ACT); out = np.full(n, -1)
    A = ACT.reshape(n, -1); B = ASW.reshape(n, -1)
    for s in np.unique(SI):
        rows = np.where(SI == s)[0]
        key = {}
        for j in rows: key.setdefault((int(GE[j]), A[j].tobytes()), j)
        for i in rows:
            out[i] = key.get((int(DGE[i]), B[i].tobytes()), -1)
    return out


def per_source(tag, reader_cache):
    d = np.load(os.path.join(RESULTS, f"{tag}_val.npz"), allow_pickle=True)
    ACT, ASW, GE, SI = d["ACT"], d["ASW"], d["GE"], d["SI"]
    DGE, SAME, F0 = d["DONOR_GE"], d["SAME"], d["F0Q"].astype(np.float32)
    DSW = d["DSW"].astype(np.float32); names = d["names"]
    Y = targets(ACT)["seq"]; YD = targets(ASW)["seq"]
    don = donor_index(ACT, ASW, GE, DGE, SI)
    res = {}
    for s in np.unique(SI):
        m = SI == s; ug = np.unique(GE[m]); rs = np.random.RandomState(1)
        te_eps = rs.choice(ug, max(1, int(len(ug) / 3)), replace=False)
        te = m & np.isin(GE, te_eps) & (SAME > 0.5) & (don >= 0); tr = m & ~np.isin(GE, te_eps)
        if tr.sum() < 20 or te.sum() < 80: continue
        if s not in reader_cache:                               # the SAME frozen reader for both arms
            DF = reader_cache["_DF"]
            reader_cache[s] = _ridge(DF[tr], Y[tr], GE[tr])[:4]
        mu, sd, W, ym = reader_cache[s]
        idx = np.where(te)[0]
        pred = standardise(DSW[idx], mu, sd) @ W + ym
        dist = np.linalg.norm(F0[idx] - F0[don[idx]], axis=1)
        q = np.quantile(dist, [0.25, 0.5, 0.75]); bins = np.digitize(dist, q)          # 0..3 = Q1..Q4
        row = {"n_test": int(len(idx)), "all": 1 - _r2(YD[idx], pred)}
        for b in range(4):
            sel = bins == b
            if sel.sum() >= 10: row[f"Q{b+1}"] = 1 - _r2(YD[idx][sel], pred[sel])
        res[str(names[s])] = row
    return res


def main():
    base = np.load(os.path.join(RESULTS, f"{ARMS['recon']}_val.npz"), allow_pickle=True)
    cache = {"_DF": base["DF"].astype(np.float32)}                # model-free: identical in every dump
    out = {arm: per_source(tag, cache) for arm, tag in ARMS.items()}
    json.dump(out, open(os.path.join(RESULTS, "transport_by_source.json"), "w"), indent=1)
    common = sorted(set(out["recon"]) & set(out["eat"]))
    for key in ("all", "Q1", "Q4"):
        pairs = [(out["recon"][s][key], out["eat"][s][key]) for s in common if key in out["recon"][s] and key in out["eat"][s]]
        a = np.array(pairs); below = int((a[:, 1] < a[:, 0]).sum())
        print(f"{key:4s}: {len(a)} sources | median recon {np.median(a[:,0]):.3f}  EAT {np.median(a[:,1]):.3f} | "
              f"EAT lower on {below}/{len(a)}", flush=True)


if __name__ == "__main__":
    main()
