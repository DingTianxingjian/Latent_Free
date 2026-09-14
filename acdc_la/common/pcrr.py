"""Portable core + reconstruction residual (review 2026-09-13).

The Stage-I posterior mean z0 = E(cells) (n_cells*d, frozen) is split by two learned heads into
    z^P = P(z0)   (d_p)   the PORTABLE core: the only thing the cross-context path may use, the only thing the policy predicts
    r   = R(z0)   (d_r)   the reconstruction RESIDUAL: allowed on the self path only, never predicted downstream
and one state-conditioned decoder realises both paths on the LATP transport architecture:
    self :  D(s_i, z^P_i, r_i)  -> dF_i            (exact reconstruction keeps its escape hatch)
    cross:  D(s_j, z^P_i, 0  )  -> dF^_{j<-i}      (what only the original clip can use must not be in z^P)
The split is defined operationally by which path may consume the information, not by a semantic TC/TI label.
Routing pressure: the swap-realism + cycle terms of Stage III act on the cross path, so donor-specific content
in z^P costs either realism (rendered on the wrong state) or cycle (not rendered); the residual is free of both.

Interface: `encode(cells) -> (zP_mu (B,n_cells,d_p/n_cells), zP_lv, r_mu (B,d_r), r_lv)` and `decode(f0, c, zP, r)`
mirror LAM, with n_cells / d / d_u = (base n_cells, d_p/n_cells, d_r), so the validation dump, every probe
(Stage I on Z = z^P, Stage II on U = r, Stage III on the swap with r = 0) and the LIBERO policy run unchanged.
"""
import torch, torch.nn as nn

from ..paths import DINO_DIM
from .model import LAM, Gauss


class PCRR(nn.Module):
    swap_u_zero = True          # dump.py: the swap / cross path zeroes the second channel

    def __init__(self, base: LAM, d_p=192, d_r=64):
        super().__init__()
        assert d_p % base.n_cells == 0, (d_p, base.n_cells)
        self.base = base
        for p in base.encoder_parameters(): p.requires_grad_(False)
        self.K, self.n_cells, self.d, self.d_u, self.d_p, self.d_r = base.K, base.n_cells, d_p // base.n_cells, d_r, d_p, d_r
        z0 = base.n_cells * base.d
        self.P, self.R = Gauss(z0, 512, d_p), Gauss(z0, 512, d_r)
        self.zp_in = nn.Linear(d_p, z0)                          # core -> the decoder's per-cell token layout
        self.r_in = nn.Linear(d_r, base.z_in.out_features)       # residual -> one decoder token (the old u slot)
        nn.init.zeros_(self.r_in.weight); nn.init.zeros_(self.r_in.bias)     # the LATP decoder was trained with u ~ 0
        self.ckpt_meta = dict(kind="pcrr", pcrr=dict(d_p=d_p, d_r=d_r, base=dict(K=base.K, n_cells=base.n_cells, d=base.d, d_u=base.d_u, layers=base.dec.num_layers)))

    def encode(self, cells):
        z0 = self.base.encode(cells)[0].flatten(1)               # frozen Stage-I posterior mean (grad flows to the input for the cycle term)
        pmu, plv = self.P(z0); rmu, rlv = self.R(z0)
        B = z0.shape[0]
        return pmu.view(B, self.n_cells, self.d), plv.view(B, self.n_cells, self.d), rmu, rlv

    def decode(self, f0, c, z, r):
        b, B = self.base, f0.shape[0]
        zt = self.zp_in(z.reshape(B, -1)).view(B, b.n_cells, b.d)
        tok = torch.cat([b.f0_in(f0), b.z_in(zt), self.r_in(r)[:, None], b.c_in(c)[:, None]], 1) + b.pos
        return b.out(b.dec(tok)[:, :16]).view(B, 16, b.K, DINO_DIM).transpose(1, 2)

    def encoder_parameters(self): return self.base.encoder_parameters()
    def freeze_encoder(self): pass                                # always frozen
