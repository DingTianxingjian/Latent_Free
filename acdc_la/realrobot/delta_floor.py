"""What does the delta arm have to beat, and what can it not beat?  (2026-11-10)

Two corrections to today's numbers, both needed before predicting anything about the delta run.

(1) The 1.49 deg bound was computed on the next ABSOLUTE target at pose-matched steps -- but the match allows the
    two poses to differ by up to 2 deg per joint, and that difference is INSIDE the absolute-target spread.  So the
    bound was inflated by the matching slack.  Recomputed on the DELTA (target minus the step's own measured
    configuration), which removes the pose offset by construction, it is the honest floor for both arms, because
    `|pred_delta - true_delta|` and `|pred_target - true_target|` are the same physical position error.

(2) The null for the delta arm.  "Output zero" means "stay where you are", so its error is mean |delta|.  This is
    the delta analogue of the persistence baseline (2.06 deg) that BEAT the absolute arm, and the delta arm has to
    get under it to be doing anything a trivial controller could not.
"""
import os, numpy as np
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf; tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds

ABS = "/home/ti874297/Projects/latent_free/data/real_rlds/piper_pick_can/1.0.0"
b = tfds.builder_from_directory(ABS)

# --- (2) the nulls, on the SAME 8 held-out episodes every other number in this ledger entry uses
A8, J8 = [], []
for epi in b.as_dataset(split="train[95%:]"):
    st = list(epi["steps"])
    A8.append(np.stack([s["action"].numpy() for s in st]).astype(np.float64))
    J8.append(np.stack([s["observation"]["joint_state"].numpy() for s in st]).astype(np.float64))
d8 = np.concatenate([a[:, :6] - j[:, :6] for a, j in zip(A8, J8)])
pers = np.concatenate([np.abs(a[1:, :6] - a[:-1, :6]) for a in A8])      # repeat the last COMMAND
print(f"held-out (8 episodes, {len(d8)} steps) -- what the delta arm must beat:")
print(f"  null 'output zero delta' = stay where you are : {np.abs(d8).mean():5.2f} deg   per joint {np.round(np.abs(d8).mean(0),2)}")
print(f"  null 'repeat the last command' (persistence)   : {pers.mean():5.2f} deg")
print(f"  the ABSOLUTE arm, measured open loop           :  3.24 deg")

# --- (1) the corrected floor: spread of the DELTA at matched states, across episodes
J, D, E, IM, PH = [], [], [], [], []
for ei, epi in enumerate(b.as_dataset(split="train")):
    st = list(epi["steps"])
    a = np.stack([s["action"].numpy() for s in st]).astype(np.float32)
    j = np.stack([s["observation"]["joint_state"].numpy() for s in st]).astype(np.float32)
    for t, s in enumerate(st):
        IM.append((tf.image.resize(tf.image.rgb_to_grayscale(s["observation"]["image"]), (32, 32)).numpy()[:, :, 0].ravel() / 255.0))
        J.append(j[t, :6]); D.append(a[t, :6] - j[t, :6]); E.append(ei); PH.append(t / (len(st) - 1))
J, D, E, IM, PH = np.stack(J), np.stack(D), np.array(E), np.stack(IM).astype(np.float32), np.array(PH)
pose_d = np.abs(J[:, None, :] - J[None, :, :]).max(2)
img_d = np.empty((len(IM), len(IM)), dtype=np.float32)
for i in range(0, len(IM), 256):
    blk = IM[i:i + 256]
    img_d[i:i + 256] = np.sqrt(np.maximum((blk**2).sum(1)[:, None] + (IM**2).sum(1)[None] - 2 * blk @ IM.T, 0) / IM.shape[1])
diff_ep = E[:, None] != E[None, :]
thr = np.quantile(img_d[diff_ep], 0.02)
ph_d = np.abs(PH[:, None] - PH[None, :])
print(f"\ncorrected floor -- spread of the DELTA at matched states, cross-episode (bound = half the spread):")
for lab, m in [("pose < 2 deg", diff_ep & (pose_d < 2.0)),
               ("pose < 2 deg + image closest 2%", diff_ep & (pose_d < 2.0) & (img_d < thr)),
               ("pose < 2 deg + image + phase<0.08", diff_ep & (pose_d < 2.0) & (img_d < thr) & (ph_d < 0.08)),
               ("pose < 1 deg + image closest 2%", diff_ep & (pose_d < 1.0) & (img_d < thr))]:
    i, j_ = np.nonzero(m)
    if not len(i): print(f"  {lab:<36} no pairs"); continue
    s = np.abs(D[i] - D[j_]).mean(1)
    print(f"  {lab:<36} pairs {len(i)//2:>7}  spread mean {s.mean():5.2f}  -> FLOOR {s.mean()/2:5.2f} deg"
          f"   (median {np.median(s):4.2f} -> {np.median(s)/2:4.2f})")
print("\n(the old 1.49 deg was the ABSOLUTE-target spread, which contains the <=2 deg pose-matching slack)")
