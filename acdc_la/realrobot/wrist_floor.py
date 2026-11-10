"""Per-joint: is there anything learnable on j3/j4/j5, or does the wrist have no signal?  (2026-11-10)

Arm B on delta data beats the zero-delta null on j0/j1/j2 (ratio 0.84/0.72/0.86) but is WORSE than it on
j3/j4/j5 (1.44/1.10/1.24).  Two very different causes, and they imply opposite next moves:

  (a) the wrist increments are unpredictable from the observation -- then the per-joint FLOOR sits at (or above)
      the per-joint null, there is nothing to learn, and a policy that outputs its conditional mean should land
      AT the null, not above it.  Outputting zero on those dims would then be the correct controller.
  (b) the wrist increments are predictable but our model is not predicting them -- the floor is well under the
      null, and the loss is simply being spent elsewhere (the degree-scale dims j1/j2 dominate an unweighted L1).

This runs the same matching as delta_floor.py but reports every quantity PER JOINT, so (a) and (b) separate.
CPU only, no GPU, no checkpoint.

  envs/univla_vla/bin/python -m acdc_la.realrobot.wrist_floor
"""
import os, numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf; tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds

ABS = "/home/ti874297/Projects/latent_free/data/real_rlds/piper_pick_can/1.0.0"
b = tfds.builder_from_directory(ABS)

# arm B's measured open-loop error, from the replay this morning (chunk[0], 353 held-out steps)
B = np.array([0.599, 2.022, 1.672, 0.419, 1.552, 0.062])

# --- the per-joint null, on the same 8 held-out episodes
d8 = []
for epi in b.as_dataset(split="train[95%:]"):
    st = list(epi["steps"])
    a = np.stack([s["action"].numpy() for s in st]).astype(np.float64)
    j = np.stack([s["observation"]["joint_state"].numpy() for s in st]).astype(np.float64)
    d8.append(a[:, :6] - j[:, :6])
d8 = np.concatenate(d8)
null = np.abs(d8).mean(0)
sd = d8.std(0)

# --- the per-joint floor: spread of the delta at matched states, across episodes, on ALL 150 episodes
J, D, E, IM = [], [], [], []
for ei, epi in enumerate(b.as_dataset(split="train")):
    st = list(epi["steps"])
    a = np.stack([s["action"].numpy() for s in st]).astype(np.float32)
    j = np.stack([s["observation"]["joint_state"].numpy() for s in st]).astype(np.float32)
    for t, s in enumerate(st):
        IM.append(tf.image.resize(tf.image.rgb_to_grayscale(s["observation"]["image"]), (32, 32)).numpy()[:, :, 0].ravel() / 255.0)
        J.append(j[t, :6]); D.append(a[t, :6] - j[t, :6]); E.append(ei)
J, D, E, IM = np.stack(J), np.stack(D), np.array(E), np.stack(IM).astype(np.float32)
pose_d = np.abs(J[:, None, :] - J[None, :, :]).max(2)
img_d = np.empty((len(IM), len(IM)), dtype=np.float32)
for i in range(0, len(IM), 256):
    blk = IM[i:i + 256]
    img_d[i:i + 256] = np.sqrt(np.maximum((blk**2).sum(1)[:, None] + (IM**2).sum(1)[None] - 2 * blk @ IM.T, 0) / IM.shape[1])
diff_ep = E[:, None] != E[None, :]
thr = np.quantile(img_d[diff_ep], 0.02)
m = diff_ep & (pose_d < 2.0) & (img_d < thr)
i, k = np.nonzero(m)
floor = np.abs(D[i] - D[k]).mean(0) / 2.0          # bound = half the spread, per joint
print(f"\n{len(i)} matched cross-episode pairs (pose < 2 deg, image closest 2%)\n")
hdr = "".join(f"{f'j{q}':>9}" for q in range(6))
print(f"{'':<34}{hdr}{'mean':>9}")
for lab, v in [("true |delta| per step  (= null)", null), ("std of the true delta", sd),
               ("FLOOR  (half the matched spread)", floor), ("arm B measured, chunk[0]", B)]:
    print(f"{lab:<34}" + "".join(f"{x:9.3f}" for x in v) + f"{v.mean():9.3f}")
print(f"\n{'ratio  floor / null':<34}" + "".join(f"{x:9.2f}" for x in floor / null))
print(f"{'ratio  arm B / null':<34}" + "".join(f"{x:9.2f}" for x in B / null))
print(f"{'ratio  arm B / floor':<34}" + "".join(f"{x:9.2f}" for x in B / floor))
print("\nreading: floor/null well BELOW 1 means the joint IS predictable from the observation, so arm B being")
print("over 1 on it is our model's shortfall, not a data limit.  floor/null at or above 1 means case (a).")
