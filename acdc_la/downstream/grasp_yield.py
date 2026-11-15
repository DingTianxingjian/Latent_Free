"""How many donors actually qualify under `collect`'s rule (first lead in 2,4,6 that ends the chunk in a grasp)?

probe only tries lead=6 and reported 59/72 donors with no grasp on libero_spatial, which would make the
`centered` stratum too thin for the pre-registered centered-vs-outside reading.  collect sweeps the lead, so
the qualification rate is the number that decides whether 1207875-78 need more demos per task.
Cheap by design: donor chunks only (<=3 runs per donor), no recipient replays.
"""
import os, sys
import numpy as np, h5py

sys.argv = ["x"]
from acdc_la.downstream.phys_replay import Sim, tasks_of, close_steps, H

for suite in os.environ.get("SUITES", "libero_spatial").split(","):
    q = l6 = n = 0; per = []
    for ti, task, f in tasks_of(suite)[:10]:
        sim = Sim(task); D = h5py.File(f, "r")["data"]
        demos = sorted(D.keys(), key=lambda s: int(s.split("_")[1]))[:8]
        qt = 0
        for di in demos:
            A, S = D[di]["actions"][()], D[di]["states"][()]
            cs = close_steps(A)
            if not cs: n += 1; continue
            tc = cs[0]; n += 1; got = None
            for lead in (2, 4, 6):
                t0 = max(0, tc - lead); sim.run(S[t0], A[t0:t0 + H], A[:t0])
                if sim.grasped(): got = lead; break
            if got is not None:
                q += 1; qt += 1
                if got == 6: l6 += 1
        per.append(qt); print(f"[yield] {suite}/{ti} {qt}/8 donors qualify  ({sim.lang[:48]})", flush=True)
        sim.env.close()
    print(f"[yield] == {suite}: {q}/{n} donors qualify with lead in (2,4,6); per task {per}", flush=True)
