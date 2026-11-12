"""Does pick-can have the same teleop LAG that bends press_button's direction at chunk >= 2?  (2026-11-12)

The press_button entry (2026-11-12) traced chunk >= 2 failures to the demos being teleoperated as a CONTINUOUS
30 Hz command: the demo arm at step s+1 is already PAST the command issued at s, so a stepwise 10 Hz arm is behind
it and the per-step delta's later entries are mis-anchored.  Derivation of the drift, data only:

  arm settles at each command and q is re-read every tick, so q_arm(s+k+1) = q_arm(s+k) + d_pred[k],
  with d_pred[k] = a[s+k] - q[s+k] (the trained per-step delta).  Writing e_k = q_arm(s+k) - a[s+k-1],
      e_{k+1} = e_k + (a[s+k-1] - q[s+k])
  so the per-step drift is exactly the demo LEAD  a[t-1] - q[t]  -- the gap between what the demonstrator commanded
  and where the arm actually was one step later.  Linear in k, which is why press_button's curve is ~+2 deg/step.

This measures that lead on pick-can, so the chunk warning for the pick-can delta checkpoints is measured, not
inherited from the other dataset.  CPU only, no model.

  envs/univla_vla/bin/python -m acdc_la.realrobot.demo_lead
"""
import os, numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf; tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds

ROOT = "/home/ti874297/Projects/latent_free/data/real_rlds/piper_pick_can/1.0.0"
lead, motion = [], []
for epi in tfds.builder_from_directory(ROOT).as_dataset(split="train"):
    st = list(epi["steps"])
    a = np.stack([s["action"].numpy() for s in st]).astype(np.float64)[:, :6]
    q = np.stack([s["observation"]["joint_state"].numpy() for s in st]).astype(np.float64)[:, :6]
    lead.append(a[:-1] - q[1:])            # what was commanded at t-1, minus where the arm actually was at t
    motion.append(q[1:] - q[:-1])          # the arm's own per-step motion, for scale
lead, motion = np.concatenate(lead), np.concatenate(motion)
hdr = "".join(f"{f'j{i}':>9}" for i in range(6))
print(f"pick-can, {len(lead)} step pairs over 150 episodes\n")
print(f"{'':<38}{hdr}{'mean|.|':>10}")
for lab, v in [("demo LEAD  a[t-1] - q[t]  (signed)", lead.mean(0)),
               ("           |lead|", np.abs(lead).mean(0)),
               ("arm motion per step  |q[t]-q[t-1]|", np.abs(motion).mean(0))]:
    print(f"{lab:<38}" + "".join(f"{x:9.3f}" for x in v) + f"{np.abs(v).mean():10.3f}")
d = np.abs(lead).mean()
print(f"\nper-step drift of a per-step-delta chunk executed open loop: {d:.2f} deg/step")
print(f"  k=1 {d:5.2f}   k=2 {2*d:5.2f}   k=4 {4*d:5.2f}   k=8 {8*d:5.2f}   k=12 {12*d:5.2f}  deg")
print(f"compare: the model's own chunk[0] error is 1.05-1.13 deg, and press_button's measured drift was ~2 deg/step")
