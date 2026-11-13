"""Is the shortfall explained by the MISSING PROPRIOCEPTION?  (stack_bowls, 2026-11-13)

wrist_floor.py's floor conditions on BOTH a matched image AND a matched pose (|dq| < 2 deg) -- but the policy is
given the image ONLY (`load_camera_views=("primary",)`, `load_proprio=False`).  So that floor is an ORACLE-STATE
floor and the policy cannot be expected to reach it.  This recomputes the same half-spread with the pose condition
REMOVED, i.e. the honest floor for a policy that sees one front frame and nothing else.

  floor_img_only ~ floor_pose+img   -> the image already pins the state down; proprioception is not the problem
  floor_img_only >> floor_pose+img  -> the image does NOT pin the state down, and part of the policy's error is
                                      irreducible for its input, not a modelling failure
"""
import argparse, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="stack_bowls")
ap.add_argument("--abs_root", default="data/real_rlds")
ap.add_argument("--measured", type=float, nargs=6, default=[0.593, 1.254, 0.918, 0.125, 1.177, 0.102],
                help="the policy's per-joint chunk[0] error, for the ratios")
ap.add_argument("--label", default="policy", help="which arm --measured belongs to")
ARG = ap.parse_args()
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf; tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds

ABS = os.path.join(ARG.abs_root, ARG.dataset, "1.0.0")
POL = np.array(ARG.measured)
print(f"[obs-floor] {ABS}   measured arm: {ARG.label}")
b = tfds.builder_from_directory(ABS)
J, D, E, IM = [], [], [], []
for ei, epi in enumerate(b.as_dataset(split="train")):
    st = list(epi["steps"])
    a = np.stack([s["action"].numpy() for s in st]).astype(np.float32)
    j = np.stack([s["observation"]["joint_state"].numpy() for s in st]).astype(np.float32)
    for t, s in enumerate(st):
        IM.append(tf.image.resize(tf.image.rgb_to_grayscale(s["observation"]["image"]), (32, 32)).numpy()[:, :, 0].ravel() / 255.0)
        J.append(j[t, :6]); D.append(a[t, :6] - j[t, :6]); E.append(ei)
J, D, E, IM = np.stack(J), np.stack(D), np.array(E), np.stack(IM).astype(np.float32)
print(f"{len(D)} steps, {E.max()+1} episodes")

pose_d = np.abs(J[:, None, :] - J[None, :, :]).max(2)
img_d = np.empty((len(IM), len(IM)), dtype=np.float32)
for i in range(0, len(IM), 256):
    blk = IM[i:i + 256]
    img_d[i:i + 256] = np.sqrt(np.maximum((blk**2).sum(1)[:, None] + (IM**2).sum(1)[None] - 2 * blk @ IM.T, 0) / IM.shape[1])
diff_ep = E[:, None] != E[None, :]
thr = np.quantile(img_d[diff_ep], 0.02)

hdr = "".join(f"{f'j{q}':>9}" for q in range(6))
print(f"\n{'':<40}{hdr}{'mean':>9}")
rows = {}
for lab, m in (("floor: image 2% AND pose < 2 deg", diff_ep & (pose_d < 2.0) & (img_d < thr)),
               ("floor: image 2% ONLY (policy's input)", diff_ep & (img_d < thr)),
               ("floor: image 0.2% ONLY", diff_ep & (img_d < np.quantile(img_d[diff_ep], 0.002))),
               ("floor: pose < 2 deg ONLY", diff_ep & (pose_d < 2.0))):
    i, k = np.nonzero(m)
    v = np.abs(D[i] - D[k]).mean(0) / 2.0
    rows[lab] = v
    print(f"{lab:<40}" + "".join(f"{x:9.3f}" for x in v) + f"{v.mean():9.3f}   ({len(i):,} pairs)")
print(f"{ARG.label + ' measured, chunk[0]':<40}" + "".join(f"{x:9.3f}" for x in POL) + f"{POL.mean():9.3f}")
for lab, v in rows.items():
    print(f"{'ratio ' + ARG.label + ' / ' + lab.split(': ')[1]:<40}" + "".join(f"{x:9.2f}" for x in POL / v) + f"{POL.mean()/v.mean():9.2f}")
