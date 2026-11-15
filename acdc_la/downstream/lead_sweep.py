"""collect anchors the donor chunk at t0 = close - lead, lead in (2,4,6), and requires a grasp at the END of the
16-step chunk.  In the slow-grasp tasks the grasp only forms ~8-12 steps AFTER the close command, so every one of
those donors is dropped -- a SYSTEMATIC loss (whole tasks), not a random one.  Does extending the sweep to negative
leads (chunk starting after the close) recover them, without touching donors that already qualify?
"""
import os, sys
import numpy as np, h5py
sys.argv = ["x"]
from acdc_la.downstream.phys_replay import Sim, tasks_of, close_steps, H

LEADS = [6, 4, 2, 0, -4, -8, -12, -16]
CASES = [("libero_spatial", 7), ("libero_spatial", 2), ("libero_goal", 0), ("libero_goal", 3),
         ("libero_goal", 5), ("libero_goal", 7), ("libero_goal", 4), ("libero_goal", 9)]
for suite, want in CASES:
    ti, task, f = [x for x in tasks_of(suite) if x[0] == want][0]
    sim = Sim(task); D = h5py.File(f, "r")["data"]
    demos = sorted(D.keys(), key=lambda s: int(s.split('_')[1]))[:8]
    old = new = 0; firsts = []
    for di in demos:
        A, S = D[di]["actions"][()], D[di]["states"][()]
        cs = close_steps(A)
        if not cs: continue
        hit = None
        for lead in LEADS:
            t0 = cs[0] - lead
            if t0 < 0 or t0 + H > len(S): continue
            sim.run(S[t0], A[t0:t0 + H], A[:t0])
            if sim.grasped(): hit = lead; break
        if hit is not None:
            new += 1; firsts.append(hit)
            if hit in (2, 4, 6): old += 1
    print(f"[sweep] {suite}/{ti:<2} old(2,4,6) {old}/8 -> extended {new}/8   first qualifying lead {firsts}   ({sim.lang[:46]})", flush=True)
    sim.env.close()
