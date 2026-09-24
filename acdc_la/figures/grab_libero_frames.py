"""Render initial-state frames from the LIBERO suites we evaluate on, for the paper's environment figure.

    MUJOCO_GL=egl python -m acdc_la.figures.grab_libero_frames --tasks 3 --res 512

Needs a GPU node (MuJoCo renders through EGL; this cluster has no OSMesa), so it runs from
acdc_la/scripts/run_libero_frames.sh.  For each suite and task it resets to the benchmark's own first
initial state, steps the same no-op settle used by the evaluator, and saves the agentview frame the
policy would see, plus the task instruction.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suites", nargs="+", default=SUITES)
    ap.add_argument("--tasks", type=int, default=3, help="tasks per suite, taken from the front of the suite")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--settle", type=int, default=10, help="no-op steps, as in the evaluator")
    ap.add_argument("--out", default="figures/libero_frames")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bench = benchmark.get_benchmark_dict()
    meta = []
    for s in a.suites:
        suite = bench[s]()
        for ti in range(min(a.tasks, suite.n_tasks)):
            task = suite.get_task(ti)
            env, desc = get_libero_env(task, "openvla", resolution=a.res)
            env.reset()
            obs = env.set_init_state(suite.get_task_init_states(ti)[0])
            for _ in range(a.settle):
                obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
            img = obs["agentview_image"][::-1]                       # LIBERO renders upside down
            p = os.path.join(a.out, f"{s}_t{ti}.png")
            Image.fromarray(np.asarray(img, dtype=np.uint8)).save(p)
            meta.append({"suite": s, "task": ti, "instruction": desc, "file": os.path.basename(p)})
            print(f"[frames] {p}  <- {desc}", flush=True)
            env.close()
    with open(os.path.join(a.out, "frames.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[frames] wrote {len(meta)} frames to {a.out}")


if __name__ == "__main__":
    main()
