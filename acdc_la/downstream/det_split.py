"""Split the probe's det number into its two parts.

probe reports |obj(replay from S[t], A[t:t+H]) - obj(restore S[t+H])|: that mixes
  (A) replay-vs-replay  -- is the sim deterministic for us?  (must be 0 after b8666cd)
  (B) replay-vs-RECORDING -- does this robosuite/LIBERO version reproduce the demo at all?
(B) > 0 with (A) == 0 means the demos are not reproducible in this env version, which is fine for the
experiment (the donor effect is then whatever the sim does, self-consistently) but makes probe's det
the wrong quantity to gate on.  (B) == 0 everywhere would mean a 17 cm outlier is still broken restore.
"""
import os, sys, glob
import numpy as np, h5py

sys.argv = ["x"]
from acdc_la.downstream.phys_replay import Sim, tasks_of, H

suite = os.environ.get("SUITE", "libero_spatial")
rows = []
for ti, task, f in tasks_of(suite)[:2]:
    sim = Sim(task); D = h5py.File(f, "r")["data"]
    demos = sorted(D.keys(), key=lambda s: int(s.split("_")[1]))[:4]
    for di in demos:
        A, S = D[di]["actions"][()], D[di]["states"][()]
        for t in range(0, len(A) - H, max(1, (len(A) - H) // 6)):
            sim.run(S[t], A[t:t + H], A[:t]); p1 = {k: v.copy() for k, v in sim.obj_pos().items()}
            sim.run(S[t], A[t:t + H], A[:t]); p2 = {k: v.copy() for k, v in sim.obj_pos().items()}
            sim.restore(S[t + H], A[:t + H]); prec = {k: v.copy() for k, v in sim.obj_pos().items()}
            a = max(np.linalg.norm(p1[n] - p2[n]) for n in p1)          # (A) replay vs replay
            b = max(np.linalg.norm(p1[n] - prec[n]) for n in p1)        # (B) replay vs recording
            rows.append((suite, ti, di, t, a * 100, b * 100))
            print(f"{suite}/{ti} {di} t={t:4d}  replay-vs-replay {a*100:8.4f} cm   replay-vs-recording {b*100:8.4f} cm", flush=True)
    sim.env.close()
A_ = np.array([r[4] for r in rows]); B_ = np.array([r[5] for r in rows])
print(f"\nn={len(rows)}")
print(f"(A) replay vs replay    : median {np.median(A_):.5f}  p95 {np.percentile(A_,95):.5f}  max {A_.max():.5f} cm   zero-fraction {(A_<1e-9).mean():.3f}")
print(f"(B) replay vs recording : median {np.median(B_):.5f}  p95 {np.percentile(B_,95):.5f}  max {B_.max():.5f} cm   zero-fraction {(B_<1e-9).mean():.3f}")
