"""Does the open-loop error grow across the action chunk?  (2026-11-10)

The labmate's first real-robot test of the piper_pick_can policy reported that the motion is right but the grasp
is imprecise, running with an action chunk of 5.  The policy was TRAINED with window_size 12 at 10 Hz, so a chunk
is 1.2 s and executing 5 of it is 0.5 s of open loop.  Nothing we have measured so far says what the error does
WITHIN a chunk -- `replay()` only looks at index 0 and at the ensembled action.  That is what this measures, on
the same 8 held-out episodes (train[95%:], never trained on):

  * joint error in DEGREES at each chunk index k, against the motion scale at that horizon (how far the target
    actually travels in k steps).  If the error at k=4 is much worse than at k=0, a shorter chunk is strictly
    better and that is a one-line change on the robot.
  * the same split into the GRASP phase (around the gripper transition) and the rest, because "understands the
    motion but the grasp is imprecise" is a claim about WHERE the error sits, not how big it is on average.
  * whether the gripper command leads or lags the demonstration, by shifting the predicted series against it.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, "/home/ti874297/Projects/latent_free")
from acdc_la.realrobot.piper_policy import PiperPolicy

CKPT = "/data/ti874297/vla_runs/tc_cont+piper_pick_can+b32x2+lr0.00035+r32+s10000+fm+q4--terra_piper"
DATA = "/home/ti874297/Projects/latent_free/data/real_rlds"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/data/ti874297/chunk_profile.json"

p = PiperPolicy(CKPT, ensemble=False)
W = p.W
import tensorflow_datasets as tfds
b = tfds.builder_from_directory(os.path.join(DATA, "piper_pick_can", "1.0.0"))
ds = b.as_dataset(split="train[95%:]")

# err[k] accumulates |pred chunk[k] - gt[t+k]|; mot[k] accumulates |gt[t] - gt[t+k]|, the motion scale at that
# horizon, so the error can be read as a fraction of the movement it is trying to produce.
err = [[] for _ in range(W)]; mot = [[] for _ in range(W)]; grip_err = [[] for _ in range(W)]
phase_err = {"approach": [], "grasp": [], "transport": [], "release": []}
pred_g, true_g = [], []

for ei, epi in enumerate(ds):
    steps = list(epi["steps"])
    imgs = [s["observation"]["image"].numpy() for s in steps]
    acts = np.stack([s["action"].numpy() for s in steps]).astype(np.float64)
    instr = steps[0]["language_instruction"].numpy().decode()
    T = len(steps)
    # The gripper transitions: biggest downward and upward jumps in the ground-truth gripper channel.
    dg = np.diff(acts[:, 6])
    t_close = int(np.argmin(dg)) if dg.size else 0      # open -> closed  == the grasp
    t_open = int(np.argmax(dg)) if dg.size else T - 1   # closed -> open  == the release
    p.reset(episode=ei)
    ep_pred_g, ep_true_g = [], []
    for t in range(T):
        ch = p.act_chunk(imgs[t], instr)                # already un-normalised: degrees + gripper in [0,1]
        ep_pred_g.append(ch[0, 6]); ep_true_g.append(acts[t, 6])
        for k in range(W):
            if t + k >= T: break
            err[k].append(np.abs(ch[k, :6] - acts[t + k, :6]).mean())
            mot[k].append(np.abs(acts[t, :6] - acts[t + k, :6]).mean())
            grip_err[k].append(abs(ch[k, 6] - acts[t + k, 6]))
        e0 = np.abs(ch[0, :6] - acts[t, :6]).mean()
        if   abs(t - t_close) <= 5: phase_err["grasp"].append(e0)
        elif abs(t - t_open) <= 5:  phase_err["release"].append(e0)
        elif t < t_close:           phase_err["approach"].append(e0)
        else:                       phase_err["transport"].append(e0)
    pred_g.append(np.array(ep_pred_g)); true_g.append(np.array(ep_true_g))
    print(f"[profile] episode {ei}: {T} steps, grasp at {t_close}, release at {t_open}", flush=True)

E = [float(np.mean(v)) for v in err]; M = [float(np.mean(v)) for v in mot]
G = [float(np.mean(v)) for v in grip_err]
print(f"\n=== joint error (deg, mean over 6 joints) by position in the 12-step chunk ===")
print("k (steps ahead)  " + "".join(f"{k:>8}" for k in range(W)))
print("our error        " + "".join(f"{v:8.2f}" for v in E))
print("motion scale     " + "".join(f"{v:8.2f}" for v in M))
print("err / motion     " + "".join(f"{(E[k]/M[k] if M[k] > 1e-9 else float('nan')):8.2f}" for k in range(W)))
print("gripper err      " + "".join(f"{v:8.3f}" for v in G))
print(f"\nmean over the first 5 (what a chunk=5 rollout executes): {np.mean(E[:5]):.2f} deg")
print(f"index 0 only    (what a chunk=1 rollout executes): {E[0]:.2f} deg")

print(f"\n=== where the error sits (chunk[0], deg) ===")
for k, v in phase_err.items():
    print(f"  {k:<10} {np.mean(v):6.2f}   n={len(v)}")

print(f"\n=== does the gripper command lead or lag? (L1 on the gripper channel, 0-1 units) ===")
best = None
for sh in range(-6, 7):
    tot, n = 0.0, 0
    for pg, tg in zip(pred_g, true_g):
        if sh >= 0: a, b_ = pg[sh:], tg[:len(tg) - sh] if sh else tg
        else:       a, b_ = pg[:len(pg) + sh], tg[-sh:]
        m = min(len(a), len(b_)); tot += np.abs(a[:m] - b_[:m]).sum(); n += m
    v = tot / max(n, 1)
    print(f"  shift {sh:+d} steps ({sh*100:+d} ms): {v:.4f}" + ("   <- best" if best is None or v < best[1] else ""))
    if best is None or v < best[1]: best = (sh, v)
print(f"  best alignment: shift {best[0]:+d} steps = {best[0]*100:+d} ms"
      f" ({'policy FIRES LATE, shifting it earlier helps' if best[0] > 0 else 'policy fires early' if best[0] < 0 else 'aligned'})")

json.dump(dict(checkpoint=CKPT, split="train[95%:]", window=W, err_by_index=E, motion_by_index=M,
               gripper_err_by_index=G, phase_err={k: float(np.mean(v)) for k, v in phase_err.items()},
               phase_n={k: len(v) for k, v in phase_err.items()}, best_gripper_shift=best[0]),
          open(OUT, "w"), indent=1)
print(f"\nwrote {OUT}")
