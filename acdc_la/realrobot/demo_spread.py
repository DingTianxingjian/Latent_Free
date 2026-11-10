"""Is the ~3 deg floor the POLICY's, or is it already in the demonstrations?  (2026-11-10)

The chunk profile left two explanations standing for a phase-independent ~3 deg precision floor: a bottleneck in
the policy, or irreducible multimodality in 150 teleop demonstrations of one task -- many valid ways to reach the
can, and the middle is not any of them.  This separates them with NO training and NO GPU, from the data alone.

Method.  Two steps from DIFFERENT episodes that are in the same arm configuration AND look the same are, to this
policy, the same input: it sees one front image and the instruction, nothing else.  If their next commanded joint
target differs by d, then no function of the policy's inputs can do better than ~d on that pair.  So the spread of
`action` over image-and-pose-matched pairs across episodes is a LOWER BOUND on any such policy's error.

Matching on the pose alone would overstate it -- two demos at the same joint angles with the can in different
places legitimately differ, and the can's position IS visible to the policy -- so the image distance is required
too, and the pose-only number is reported next to it as the control that shows how much the image constrains it.
"""
import os, sys, numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds

b = tfds.builder_from_directory("/home/ti874297/Projects/latent_free/data/real_rlds/piper_pick_can/1.0.0")
J, A, E, IM, PH = [], [], [], [], []
for ei, epi in enumerate(b.as_dataset(split="train")):
    steps = list(epi["steps"])
    acts = np.stack([s["action"].numpy() for s in steps]).astype(np.float32)
    for t, s in enumerate(steps):
        img = tf.image.resize(tf.image.rgb_to_grayscale(s["observation"]["image"]), (32, 32)).numpy()[:, :, 0]
        IM.append(img.ravel() / 255.0)
        # the arm's configuration at time t is the target it was commanded at t-1 (absolute joint targets)
        J.append(acts[t - 1, :6] if t else acts[0, :6]); A.append(acts[t]); E.append(ei); PH.append(t / (len(steps) - 1))
J, A, E, IM, PH = np.stack(J), np.stack(A), np.array(E), np.stack(IM).astype(np.float32), np.array(PH)
print(f"{len(J)} steps, {E.max()+1} episodes, image descriptor {IM.shape[1]}-d", flush=True)

# pose distance (max over joints, degrees) and image distance (rms over the 32x32 grey descriptor)
POSE_TOL, IMG_Q = 2.0, 0.02      # same configuration to within 2 deg on every joint; image in the closest 2 %
pose_d = np.abs(J[:, None, :] - J[None, :, :]).max(2)
img_d = np.sqrt(((IM[:, None, :] - IM[None, :, :]) ** 2).mean(2)) if len(IM) < 3000 else None
if img_d is None:                                   # 6.5k x 6.5k x 1024 would be 170 GB; do it in row blocks
    img_d = np.empty((len(IM), len(IM)), dtype=np.float32)
    for i in range(0, len(IM), 256):
        blk = IM[i:i + 256]
        img_d[i:i + 256] = np.sqrt(np.maximum((blk ** 2).sum(1)[:, None] + (IM ** 2).sum(1)[None]
                                              - 2 * blk @ IM.T, 0) / IM.shape[1])
diff_ep = E[:, None] != E[None, :]
thr = np.quantile(img_d[diff_ep], IMG_Q)
print(f"image distance: median across episodes {np.median(img_d[diff_ep]):.4f}, {IMG_Q:.0%} quantile {thr:.4f}")

def spread(m, label):
    i, j = np.nonzero(m)
    if not len(i): print(f"  {label}: no pairs"); return
    d = np.abs(A[i, :6] - A[j, :6]).mean(1)
    # LOWER BOUND on the mean error of ANY deterministic policy of these inputs: |f-a1|+|f-a2| >= |a1-a2|,
    # so the best it can do on a matched pair averages HALF the pair's disagreement.
    print(f"  {label:<46} pairs {len(i)//2:>7}   spread mean {d.mean():5.2f} median {np.median(d):5.2f}"
          f"   -> BOUND {d.mean()/2:5.2f} (median {np.median(d)/2:4.2f})  deg")

print(f"\nspread of the next commanded joint target over matched pairs from DIFFERENT episodes:")
spread(diff_ep & (pose_d < POSE_TOL), f"pose < {POSE_TOL} deg only (control)")
spread(diff_ep & (pose_d < POSE_TOL) & (img_d < thr), f"pose < {POSE_TOL} deg AND image in closest {IMG_Q:.0%}")
spread(diff_ep & (img_d < thr), f"image in closest {IMG_Q:.0%} only")
ph_d = np.abs(PH[:, None] - PH[None, :])
print("  --- adding the task-phase constraint: the same configuration is passed on the way out AND back, so a")
print("      pose+image match alone can pair up different MOMENTS of the task, which inflates the spread ---")
for pt in (0.15, 0.08):
    spread(diff_ep & (pose_d < POSE_TOL) & (img_d < thr) & (ph_d < pt), f"pose+image, |phase diff| < {pt}")
    spread(diff_ep & (pose_d < POSE_TOL) & (ph_d < pt), f"pose only,  |phase diff| < {pt}")
print(f"\nfor reference: our policy's open-loop error 3.24 deg, persistence 2.06 deg, average demo 11.02 deg")
