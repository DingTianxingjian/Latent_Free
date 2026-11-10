"""Per-phase motion scale from the ground truth alone -- no policy, no GPU.

The chunk profile reported chunk[0] joint error by phase (approach 2.89, grasp 2.94, transport 4.34,
release 2.34 deg).  Reading "the grasp is not the worst phase" off that is confounded: the arm moves at
different speeds in different phases, and a slow phase gets a low absolute error for free.  This divides
by the per-step motion in each phase so the phases can actually be compared.
"""
import os, numpy as np, tensorflow_datasets as tfds
b = tfds.builder_from_directory("/home/ti874297/Projects/latent_free/data/real_rlds/piper_pick_can/1.0.0")
mot = {"approach": [], "grasp": [], "transport": [], "release": []}
for ei, epi in enumerate(b.as_dataset(split="train[95%:]")):
    acts = np.stack([s["action"].numpy() for s in epi["steps"]]).astype(np.float64)
    T = len(acts); dg = np.diff(acts[:, 6])
    t_close, t_open = int(np.argmin(dg)), int(np.argmax(dg))
    for t in range(1, T):
        m = np.abs(acts[t, :6] - acts[t - 1, :6]).mean()
        if   abs(t - t_close) <= 5: mot["grasp"].append(m)
        elif abs(t - t_open) <= 5:  mot["release"].append(m)
        elif t < t_close:           mot["approach"].append(m)
        else:                       mot["transport"].append(m)
err = {"approach": 2.89, "grasp": 2.94, "transport": 4.34, "release": 2.34}
print(f"{'phase':<11}{'err deg':>9}{'motion/step':>13}{'err/motion':>12}{'n':>6}")
for k in ("approach", "grasp", "transport", "release"):
    m = float(np.mean(mot[k]))
    print(f"{k:<11}{err[k]:9.2f}{m:13.2f}{err[k]/m:12.2f}{len(mot[k]):6d}")
