"""One place that turns a MixDS batch into what every stage consumes.

    batch = (clip u8 (B,T,H,W,3), global_ep, lang_row, act (B,K,7), langvec (B,768), src)
    ->  g4 (B,T,16,768)   frozen DINOv2 4x4 grid per frame
        dF (B,K,16,768)   unit-scaled feature differences
        f0 (B,16,768)     recipient state tokens (first frame);  c (B,768) pooled-T5 instruction
        cells (B,n_cells,768)   the tokenizer input under the chosen pooling (see common.features)
        act (B,K,3) xyz per segment, ge, si

`Featurizer.cells_of(dF)` re-applies the same pooling to any (B,K,16,768) tensor -- decoded outputs
included -- so the cycle term and the readers never see a different operator than training did.
"""
import torch

from ..paths import DINO_DIM
from . import features as Fz
from ..data.distractor import clip_to_frames


class Featurizer:
    def __init__(self, dino, K, pool="acdc", n_modes=2, dstats=1.0, dev="cuda"):
        self.dino, self.K, self.pool, self.n_modes, self.dev = dino, K, pool, n_modes, dev
        self.dstats = torch.as_tensor(float(dstats), device=dev)
        self.basis = Fz.dct_basis(K, n_modes, dev) if pool == "acdc" else None
        self.n_cells = 4 * n_modes if pool == "acdc" else 4

    def cells_of(self, dF):
        dq = Fz.quadrants(dF)
        return Fz.effect_cells(dq, self.basis) if self.pool == "acdc" else Fz.pooled_cells(dq, self.pool)

    def cells_reversed(self, dF):
        """Cells of the time-reversed clip (dF_k -> -dF_{K-1-k})."""
        return self.cells_of(-dF.flip(1))

    @torch.no_grad()
    def g4_of(self, frames):
        return Fz.clip_g4(self.dino, frames)

    def __call__(self, batch, distractor=None):
        clip, ge, lrow, act, lv, si = batch
        frames = clip_to_frames(clip, self.dev)
        out = dict(ge=ge.to(self.dev), si=si, act=act[:, :, :3], lrow=lrow, frames=frames)
        g4 = self.g4_of(frames)
        out["g4"], out["dF"], out["f0"], out["c"] = g4, Fz.dF_from_g4(g4, self.dstats), g4[:, 0], lv.to(self.dev)
        out["cells"] = self.cells_of(out["dF"])
        if distractor is not None:                                   # the SAME clips with the moving patch
            fp = distractor.render(frames, ge)
            g4p = self.g4_of(fp)
            out["dF_patch"] = Fz.dF_from_g4(g4p, self.dstats); out["f0_patch"] = g4p[:, 0]
            out["cells_patch"] = self.cells_of(out["dF_patch"])
        return out
