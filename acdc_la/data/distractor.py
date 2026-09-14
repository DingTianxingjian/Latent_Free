"""Moving-patch nuisance: a 64x64 natural-video crop travelling around the frame border.

This is the exogenous-motion INTERVENTION every stage reads against: the same clip is rendered
clean and with the patch, the patch motion is independent of the robot, and a code that is
invariant to it (Stage I "patch drop", Stage II N_TI vs N_TC) is measured, not asserted.
The rendering is `latent_free/features.py::apply_distractor(cond="patch")` byte-for-byte, so
numbers are comparable to every earlier readout in the ledger; this module only fixes the
paths and exposes the two entry points the stages need.
"""
import os, sys
import torch
import torch.nn.functional as F

from ..paths import LF, IMG, DISTRACTOR_VIDEO

sys.path.insert(0, LF)
import features as _lf_features            # noqa: E402  latent_free/features.py (frozen; diagnostics depend on it)


class Distractor:
    """Holds the video and border mask on `dev`; `render(clip, ep_id)` composites the moving patch."""

    def __init__(self, dev):
        self.video = _lf_features.load_video(dev)
        self.bmask, self.b = _lf_features.border_mask(dev)

    def render(self, frames, ep_ids, cond="patch"):
        """frames (B,T,3,IMG,IMG) float in [0,1]; ep_ids (B,) ints (seed the patch trajectory per
        episode, so the nuisance is temporally coherent within an episode) -> same shape."""
        return torch.stack([_lf_features.apply_distractor(frames[i], cond, int(ep_ids[i]), self.video, self.bmask, self.b)
                            for i in range(frames.shape[0])])


def clip_to_frames(clip_u8, dev):
    """(B,T,H,W,3) uint8 -> (B,T,3,IMG,IMG) float in [0,1] (bilinear resize, as every probe in the ledger)."""
    B, T = clip_u8.shape[:2]
    x = clip_u8.to(dev, non_blocking=True).reshape(B * T, *clip_u8.shape[2:]).permute(0, 3, 1, 2).float().div_(255.)
    return F.interpolate(x, size=(IMG, IMG), mode="bilinear", align_corners=False).view(B, T, 3, IMG, IMG)


def patch_trajectory(ep_ids, T, stride_frames=1):
    """Nuisance TARGET for Stage II: the moving patch's centre (x, y) per clip frame, in pixels, from the
    same deterministic law `apply_distractor` renders (`episode_distractor_params`, `perimeter_topleft`).
    Returns (B,T,2).  The per-segment displacement (B,T-1,2) is the nuisance analogue of the action."""
    import numpy as np
    out = np.zeros((len(ep_ids), T, 2), np.float32)
    n_video = np.load(DISTRACTOR_VIDEO, mmap_mode="r").shape[0]   # v0 = randint(n_video) is drawn FIRST and its
    # rejection sampling depends on n_video, so the true length (200) is needed for u0/speed to match the render
    for i, e in enumerate(ep_ids):
        p = _lf_features.episode_distractor_params(int(e), n_video)
        for t in range(T):
            x, y = _lf_features.perimeter_topleft(p["u0"] + p["speed"] * t)
            out[i, t] = (x + _lf_features.PATCH_SIZE / 2, y + _lf_features.PATCH_SIZE / 2)
    return out
