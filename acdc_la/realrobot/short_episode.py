"""What does a SHORT episode produce?  (2026-11-10, for the press_button handoff.)

press_button episodes are 38-167 frames at 30 Hz, i.e. 12-55 steps at 10 Hz, against a 14-frame sample window
(RLDSDataset is built with window_size = cfg.window_size + 2 = 14).  This runs prismatic's real chunker on a
synthetic 12-step trajectory to see exactly what comes out.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, tensorflow as tf
tf.config.set_visible_devices([], "GPU")
import sys
sys.path[:0] = ["/home/ti874297/Projects/latent_free", "/home/ti874297/Projects/latent_free/third_party/UniVLA"]
from prismatic.vla.datasets.rlds import traj_transforms as tt

W = 14                                        # cfg.window_size + 2
for T in (12, 17, 43):
    traj = {"observation": {"image_primary": tf.constant(np.arange(T, dtype=np.int32)[:, None, None, None]
                                                         * np.ones((1, 2, 2, 3), np.int32))},
            "action": tf.constant(np.arange(T, dtype=np.float32)[:, None] * np.ones((1, 7), np.float32)), "task": {}}
    out = tt.chunk_act_obs_libero({k: (dict(v) if isinstance(v, dict) else v) for k, v in traj.items()}, W)
    img = out["observation"]["image_primary"].numpy()[:, :, 0, 0, 0]   # the frame INDEX stored in each pixel
    act = out["action"].numpy()[:, :, 0]                                # the action INDEX
    pad = out["observation"]["pad_mask"].numpy()
    # the ctrl transform feeds the VLA imgs[ro], ro in {0,1}, and the labeller reads ro + k*3 for k=0..4
    print(f"\n--- episode of T={T} steps, window {W} ---")
    print(f"  samples produced: {img.shape[0]}  (one per timestep, none dropped)")
    for t in (0, 5, 13, T - 1):
        if t >= T: continue
        print(f"  t={t:2}: obs frame idx {list(img[t])}")
        print(f"        action idx    {[int(v) for v in act[t]]}")
        print(f"        pad_mask      {[int(v) for v in pad[t]]}  ({(~pad[t]).sum()} padded)")
    full = (pad.all(1)).sum()
    print(f"  samples with a FULLY real window: {full}/{T} = {100*full/T:.0f}%")
    print(f"  -> the ctrl transform's obs is window index 0 or 1, and the labeller spans index ro..ro+12,")
    print(f"     so a sample is degenerate whenever the window is padded at all.")
