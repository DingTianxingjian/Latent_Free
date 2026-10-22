"""CTRL_OBS_KEEP must not change a single label (2026-10-16).

(1) Pipeline: chunk_act_obs_libero with CTRL_OBS_KEEP returns exactly the kept columns of the full 14-frame window
    (observations), and the full action chunk unchanged.
(2) Transform: for ro = 0 and ro = 1, the frames the tc_cont labeller and the policy image read through KeptFrames are
    byte-identical to the ones they read from the full window, and the labeller's encode inputs are identical.

    CTRL_OBS_KEEP=0,1,3,4,6,7,9,10,12,13 python acdc_la/scripts/test_obs_keep.py

T (frames per label) is read from the REAL labeller checkpoint (T = K + 1), never assumed: the first version assumed T=4
and passed while latp2_inv has T=5 -- the KeptFrames guard caught it in the 2026-10-17 smoke (frame 7 requested).
"""
import os, sys, importlib, random
import numpy as np

KEEP = os.environ.get("CTRL_OBS_KEEP", "0,1,3,4,6,7,9,10,12,13")
LF = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [LF, os.path.join(LF, "third_party", "UniVLA")]

# ---- (1) pipeline
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
from prismatic.vla.datasets.rlds import traj_transforms as tt
T, W = 40, 14
rng = np.random.default_rng(0)
traj = lambda: {"observation": {"image_primary": tf.constant(rng.integers(0, 255, (T, 6, 6, 3), dtype=np.uint8))},
                "action": tf.constant(np.arange(T * 7, dtype=np.float32).reshape(T, 7)), "task": {}}
base = traj()
os.environ.pop("CTRL_OBS_KEEP", None)
full = tt.chunk_act_obs_libero({k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}, W)
os.environ["CTRL_OBS_KEEP"] = KEEP
kept = tt.chunk_act_obs_libero({k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}, W)
keep = [int(k) for k in KEEP.split(",")]
fo, ko = full["observation"]["image_primary"].numpy(), kept["observation"]["image_primary"].numpy()
assert ko.shape == (T, len(keep), 6, 6, 3), ko.shape
assert np.array_equal(ko, fo[:, keep]), "kept observations differ from the full window's columns"
assert np.array_equal(kept["action"].numpy(), full["action"].numpy()), "actions changed"
assert np.array_equal(kept["observation"]["pad_mask"].numpy(), full["observation"]["pad_mask"].numpy()[:, keep])
print(f"(1) pipeline OK: {T} timesteps, window {W} -> kept {keep}; observations identical, actions identical")

# ---- (2) transform
os.environ["CTRL_OBS_KEEP"] = KEEP
import acdc_la.downstream.vla.datasets_ctrl as dc
importlib.reload(dc)


import torch
TAG = os.environ.get("TC_TAG", "latp2_inv")
_ck = torch.load(os.path.join(LF, "acdc_la", "ckpt", f"{TAG}.pt"), map_location="cpu", weights_only=False)
T_REAL = int(_ck["K"]) + 1                 # TCVQLatentTokenizer: self.T = self.tok.K + 1
print(f"labeller {TAG}: K={_ck['K']} -> T={T_REAL}, stride={12 // (T_REAL - 1)}")


class RecordingLabeller:                 # stands in for TCVQLatentTokenizer: records exactly what it is asked to encode
    T = T_REAL
    def __init__(self): self.calls = []
    def encode_continuous(self, frames):
        self.calls.append([np.asarray(f).copy() for f in frames]); return np.zeros(4, dtype=np.float32)


def labeller_inputs(imgs_window, ro, use_keep):
    lab = RecordingLabeller()
    t = dc.RLDSBatchTransformCtrl.__new__(dc.RLDSBatchTransformCtrl)
    t.labeller, t.latent, t.window_size = lab, "tc_cont", 12
    imgs = dc.KeptFrames(imgs_window[keep], keep) if use_keep else imgs_window
    t.tokens(imgs, ro); policy = np.asarray(imgs[ro]).copy()
    if ro > 0: t.tokens(imgs, 0)
    return lab.calls, policy


win = rng.integers(0, 255, (W, 8, 8, 3), dtype=np.uint8)
for ro in (0, 1):
    a, pa = labeller_inputs(win, ro, False)
    b, pb = labeller_inputs(win, ro, True)
    assert len(a) == len(b) and all(len(x) == len(y) and all(np.array_equal(u, v) for u, v in zip(x, y)) for x, y in zip(a, b)), ro
    assert np.array_equal(pa, pb), ro
    print(f"(2) transform OK for ro={ro}: {sum(len(x) for x in a)} labeller frames + the policy image identical")
try:
    dc.KeptFrames(win[keep], keep)[2]; raise SystemExit("KeptFrames returned a frame that was not kept")
except KeyError:
    print("(3) asking for a dropped frame raises KeyError (no silent substitute)")
print("ALL OBS_KEEP TESTS PASSED")
