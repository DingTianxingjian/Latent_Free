"""How fast is one act() call?  (2026-11-10)

The chunk profile says executing chunk[0] only (re-plan every step) costs 3.24 deg of commanded joint error
against 4.38 deg for the first five, so a shorter chunk is better -- but only if inference keeps up with the
10 Hz the data was subsampled to.  At chunk = n the budget per forward is n * 100 ms.  Measured here on the
workstation GPU; a different card scales roughly with its bf16 throughput.
"""
import sys, time
import numpy as np
sys.path.insert(0, "/home/ti874297/Projects/latent_free")
from acdc_la.realrobot.piper_policy import PiperPolicy

p = PiperPolicy("/data/ti874297/vla_runs/tc_cont+piper_pick_can+b32x2+lr0.00035+r32+s10000+fm+q4--terra_piper",
                ensemble=False)
img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)       # a camera-shaped frame
instr = "pick up the can and place it on the plate"
p.reset()
for _ in range(3): p.act_chunk(img, instr)                        # warm up kernels / autotune
ts = []
for _ in range(30):
    t0 = time.perf_counter(); p.act_chunk(img, instr); ts.append((time.perf_counter() - t0) * 1000)
ts = np.array(ts)
import subprocess
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip().splitlines()[0]
print(f"\nGPU: {gpu}")
print(f"act_chunk: mean {ts.mean():.1f} ms, median {np.median(ts):.1f}, p90 {np.percentile(ts,90):.1f}, max {ts.max():.1f}")
print(f"10 Hz budget per chunk of n:  n=1 -> 100 ms, n=2 -> 200 ms, n=5 -> 500 ms")
for n in (1, 2, 3, 5):
    ok = "FITS" if np.percentile(ts, 90) < n * 100 else "does NOT fit"
    print(f"  chunk={n}: {n*100} ms budget, p90 {np.percentile(ts,90):.0f} ms -> {ok}")
