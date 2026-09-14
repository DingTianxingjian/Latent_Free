"""Frozen features and the fixed spatial x temporal basis (Stage I's "preserve" operator).

A clip of T frames gives K = T-1 feature differences dF_k = g4(F_{k+1}) - g4(F_k), each a 4x4
grid of 768-d DINOv2 patch means.  The tokenizer never sees the raw sequence: it sees EFFECT
CELLS, one per (spatial quadrant, temporal mode):

    cell[q, c] = sum_k  b_c[k] * mean_{p in quadrant q} dF_k[p]          (768-d each)

with b_0 = DC = all-ones (the SUM over segments = net displacement, extensive, so it composes
additively over sub-clips) and b_m = AC_m = orthonormal DCT-II rows (trajectory shape).  What
the alternatives throw away is exactly what the Stage-I probes measure:
    mean-pooled   b = 1/K   ->  DC only, scaled (UniVLA's T=2 pair IS this with K=1)
    endpoint      F_T - F_0 ->  DC only
    DC+AC1        our default (8 cells: 4 quadrants x 2 modes)
    DC+AC1+AC2    one more shape mode per quadrant
Time reversal maps dF_k -> -dF_{K-1-k}; a mode with reversal parity s_c (b_c[K-1-k] = s_c b_c[k])
therefore transforms with sign -s_c: DC (s=+1) flips sign, AC1 (s=-1) is invariant, AC2 flips.
`reversal_signs` returns these so the time-reversal probe is derived from the basis, not typed in.
"""
import math, os
import numpy as np
import torch

from ..paths import TORCH_HUB, IMG, DINO_DIM

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_dino(dev):
    torch.hub.set_dir(TORCH_HUB)
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg", source="github", trust_repo=True)
    return m.to(dev).eval().requires_grad_(False)


@torch.no_grad()
def dino_g4(dino, x):
    """x (N,3,IMG,IMG) in [0,1] -> (N,16,768): 16x16 patch tokens mean-pooled to a 4x4 grid."""
    x = (x - IMNET_MEAN.to(x.device)) / IMNET_STD.to(x.device)
    f = dino.forward_features(x)["x_norm_patchtokens"]                        # (N,256,768)
    f = f.view(-1, 16, 16, DINO_DIM).view(-1, 4, 4, 4, 4, DINO_DIM).mean((2, 4))
    return f.reshape(-1, 16, DINO_DIM)


@torch.no_grad()
def clip_g4(dino, frames):
    """frames (B,T,3,IMG,IMG) -> g4 (B,T,16,768), bf16 autocast on cuda."""
    B, T = frames.shape[:2]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=frames.is_cuda):
        return dino_g4(dino, frames.reshape(B * T, *frames.shape[2:])).float().view(B, T, 16, DINO_DIM)


def quadrants(g4):
    """(...,16,768) 4x4 grid -> (...,4,768) 2x2 quadrant means."""
    s = g4.shape[:-2]
    return g4.reshape(*s, 2, 2, 2, 2, DINO_DIM).mean((-4, -2)).reshape(*s, 4, DINO_DIM)


def dct_basis(K, n_modes=2, dev="cpu"):
    """(n_modes, K): row 0 = DC (all ones, the SUM), rows m>=1 = orthonormal DCT-II AC_m."""
    k = torch.arange(K, dtype=torch.float32, device=dev)
    rows = [torch.ones(K, device=dev)]
    for m in range(1, n_modes):
        rows.append(math.sqrt(2.0 / K) * torch.cos(math.pi * m * (k + 0.5) / K))
    return torch.stack(rows)


def reversal_signs(basis):
    """Sign each mode's cell picks up under clip reversal (dF_k -> -dF_{K-1-k}): -parity(b_c)."""
    flipped = basis.flip(1)
    par = torch.sign((basis * flipped).sum(1))                                 # +1 even, -1 odd
    assert torch.allclose(flipped, par[:, None] * basis, atol=1e-5), "basis rows must have definite parity"
    return -par


def effect_cells(dq, basis):
    """dq (B,K,4,768) quadrant deltas -> (B, 4*n_modes, 768), cells ordered [q0m0,q0m1,..,q1m0,...]."""
    c = torch.einsum("ck,bkqd->bqcd", basis, dq)                              # (B,4,C,768)
    return c.reshape(dq.shape[0], -1, DINO_DIM)


def pooled_cells(dq, kind):
    """Stage-I ablation inputs on the same quadrant deltas dq (B,K,4,768):
       mean      : (B,4,768) K^-1 sum_k dq_k        (UniVLA's pair at K=1; DC up to scale)
       endpoint  : (B,4,768) sum_k dq_k             (F_T - F_0; identical to DC)
       Both are 4 cells so the tokenizer sees them as 4 x 1 mode."""
    if kind == "mean": return dq.mean(1)
    if kind == "endpoint": return dq.sum(1)
    raise ValueError(kind)


def dF_from_g4(g4, dstats):
    """(B,T,16,768) -> unit-scaled differences (B,K,16,768)."""
    return (g4[:, 1:] - g4[:, :-1]) / dstats


@torch.no_grad()
def estimate_dstats(dino, dl, dev, n_batches=8):
    """RMS of raw dF over the first batches: a constant of the frozen features, saved with every ckpt."""
    from ..data.distractor import clip_to_frames
    sq, n = 0.0, 0
    for batch in dl:
        g4 = clip_g4(dino, clip_to_frames(batch[0], dev))
        sq += (g4[:, 1:] - g4[:, :-1]).pow(2).mean().item(); n += 1
        if n == n_batches: break
    return math.sqrt(sq / n)
