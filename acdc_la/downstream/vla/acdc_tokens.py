"""Our tokenizer as a latent-action LABELLER for UniVLA's LIBERO fine-tuning: T frames -> n_cells discrete tokens.
The code z = E(effect cells of the frozen DINO g4 dynamics) of `s1_acdc_1ep` is quantised per cell with a k-means
codebook fitted on the OXE validation dump (ckpt/vq_s1_acdc_1ep_K{Kc}.npz), so the VLA predicts 8 `<ACT_i>` tokens
of a Kc-word vocabulary where UniVLA predicts 4 tokens of 16.  Nothing here is fitted on LIBERO."""
import os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from ...paths import CKPT, IMG
from ...common import features as Fz
from ...common.model import load_ckpt, legacy_path
from ...common.pipeline import Featurizer


class ACDCLatentTokenizer(nn.Module):
    def __init__(self, tag="s1_acdc_1ep", codebook="vq_s1_acdc_1ep_K64.npz", dev="cuda"):
        super().__init__()
        p = os.path.join(CKPT, f"{tag}.pt"); self.tok, dstats, _ = load_ckpt(p if os.path.exists(p) else legacy_path(tag), dev)
        self.tok.eval().requires_grad_(False)
        self.dino = Fz.load_dino(dev); self.feat = Featurizer(self.dino, self.tok.K, "acdc", self.tok.n_cells // 4, dstats, dev)
        cb = np.load(os.path.join(CKPT, codebook)); self.register_buffer("C", torch.as_tensor(cb["C"], dtype=torch.float32).to(dev))   # (n_tokens, Kc, d_sub)
        self.n_sub = int(cb["n_sub"]) if "n_sub" in cb else 1                       # product quantisation: n_sub sub-vectors per cell
        self.n_tokens, self.codebook_size, self.T, self.dev = self.tok.n_cells * self.n_sub, int(cb["Kc"]), self.tok.K + 1, dev
        assert self.C.shape[0] == self.n_tokens and self.C.shape[2] * self.n_sub == self.tok.d, (self.C.shape, self.tok.n_cells, self.tok.d)

    @property
    def device(self): return self.dev

    @torch.no_grad()
    def encode(self, frames):
        """frames: sequence of T uint8 images (H,W,3) -> LongTensor (n_cells,) of code indices."""
        x = torch.as_tensor(np.stack([np.asarray(f) for f in frames])).to(self.dev).permute(0, 3, 1, 2).float().div_(255.)
        x = F.interpolate(x, size=(IMG, IMG), mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(self.dev == "cuda")): g4 = Fz.dino_g4(self.dino, x).float()   # (T,16,768)
        dF = ((g4[1:] - g4[:-1]) / self.feat.dstats)[None]                       # (1,K,16,768)
        z = self.tok.encode(self.feat.cells_of(dF))[0][0].reshape(self.n_tokens, -1)   # (n_tokens, d_sub) posterior mean, cells split into sub-vectors
        return ((z[:, None, :] - self.C) ** 2).sum(-1).argmin(-1)
