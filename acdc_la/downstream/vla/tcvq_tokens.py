"""The structured tokenizer (common/tcvq.py) as a labeller for the controlled VLA fine-tuning.

Handles both bottlenecks with one class:
    bottleneck "vq"          -> `encode(frames)` returns n_tokens indices of a `codebook`-entry vocabulary
    bottleneck "continuous"  -> `encode_continuous(frames)` returns the standardised n_tokens*latent_dim code
The window is sub-sampled to the tokenizer's T frames at a uniform stride, exactly as `acdc_tokens.py` does, so the
label describes the same span of the 12-step chunk the action head predicts.
"""
import os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from ...paths import CKPT, IMG, RESULTS
from ...common import features as Fz
from ...common.model import load_ckpt


class TCVQLatentTokenizer(nn.Module):
    def __init__(self, tag="tc_cont", dev="cuda"):
        super().__init__()
        self.tok, self.dstats, _ = load_ckpt(os.path.join(CKPT, f"{tag}.pt"), dev)
        self.tok.eval().requires_grad_(False)
        self.dino = Fz.load_dino(dev)
        self.T, self.dev, self.tag = self.tok.K + 1, dev, tag
        self.n_tokens = self.tok.n_tokens
        self.codebook_size = self.tok.codebook_size if self.tok.vq is not None else 0
        self.dz = self.tok.n_tokens * self.tok.latent_dim if self.tok.vq is None else 0
        p = os.path.join(CKPT, f"z_stats_{tag}.npz")                      # standardisation for the continuous head
        if self.dz and os.path.exists(p):
            zs = np.load(p); self.register_buffer("zmean", torch.as_tensor(zs["mean"]).to(dev)); self.register_buffer("zstd", torch.as_tensor(zs["std"]).to(dev))
        elif self.dz:
            self.register_buffer("zmean", torch.zeros(self.dz, device=dev)); self.register_buffer("zstd", torch.ones(self.dz, device=dev))
            print(f"[tcvq-labeller] WARNING: no z_stats_{tag}.npz; the continuous code is fed unstandardised", flush=True)

    @property
    def device(self): return self.dev

    @torch.no_grad()
    def _code(self, frames):
        x = torch.as_tensor(np.stack([np.asarray(f) for f in frames])).to(self.dev).permute(0, 3, 1, 2).float().div_(255.)
        x = F.interpolate(x, size=(IMG, IMG), mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(self.dev != "cpu")): g4 = Fz.dino_g4(self.dino, x).float()
        f0 = g4[:1]; dF = ((g4[1:] - g4[:-1]) / self.dstats)[None]        # (1,16,768) and (1,K,16,768)
        return self.tok.quantize(f0, dF)

    @torch.no_grad()
    def encode(self, frames):
        return self._code(frames)[3].reshape(-1)

    @torch.no_grad()
    def encode_continuous(self, frames):
        z = self._code(frames)[0].reshape(-1)
        return (z - self.zmean) / self.zstd


def write_z_stats(tag):
    """After training, standardise the continuous code by the statistics of its own validation dump."""
    d = np.load(os.path.join(RESULTS, f"{tag}_val.npz"), allow_pickle=True); Z = d["Z"].astype(np.float64)
    np.savez(os.path.join(CKPT, f"z_stats_{tag}.npz"), mean=Z.mean(0).astype(np.float32), std=(Z.std(0) + 1e-6).astype(np.float32),
             dim=Z.shape[1], source=f"results/{tag}_val.npz")
    print(f"[tcvq-labeller] z stats for {tag}: dim {Z.shape[1]}, mean|.| {np.abs(Z.mean(0)).mean():.4f}, std {Z.std(0).mean():.4f}")


if __name__ == "__main__":
    import sys
    for t in sys.argv[1:]: write_z_stats(t)
