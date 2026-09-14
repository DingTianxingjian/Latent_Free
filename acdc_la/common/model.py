"""Shared modules: the Gaussian cell tokenizer (E), the nuisance head (U), the state-conditioned
transformer decoder (the transport operator of Stage III), and the loss helpers.

Attribute names are the legacy ones (`E, U, cell_emb, z_in, u_in, c_in, f0_in, pos, dec, out`) so
every `claude_survey/ckpt/swaplam_*.pt` loads with `load_legacy` and the Stage-III result of the
ledger (rec_src tokenizer -> swap_fz operator) is reproducible from this repo without retraining
Stage I.  `n_cells` generalises the fixed 8 (4 quadrants x {DC, AC1}); legacy checkpoints have 8.
"""
import math, os
import torch, torch.nn as nn, torch.nn.functional as F

from ..paths import DINO_DIM, T5_DIM, LEGACY_CKPT


class Gauss(nn.Module):
    def __init__(self, i, h, o):
        super().__init__(); self.net = nn.Sequential(nn.LayerNorm(i), nn.Linear(i, h), nn.GELU(), nn.Linear(h, 2 * o))
    def forward(self, x):
        mu, lv = self.net(x).chunk(2, -1); return mu, lv.clamp(-8, 4)


class LAM(nn.Module):
    """Tokenizer + transport decoder.
       encode(cells)        : Stage-I tokenizer.  cells (B,n_cells,768) -> z (B,n_cells,d) per cell (shared E plus a
                              learned cell embedding) and a global nuisance residual u (B,d_u).  z is the only thing
                              Stage III ever swaps; u never is (that asymmetry defines the channels).
       decode(f0, c, z, u)  : T(s, z; u): recipient state s = (first-frame g4 tokens f0, pooled-T5 instruction c),
                              effect code z, residual u -> predicted dF^ (B,K,16,768).  In Stage I this is the
                              reconstruction decoder; in Stage III, with the encoder frozen, it is the transport operator."""
    def __init__(self, K, n_cells=8, d=32, d_u=8, layers=4, width=DINO_DIM, heads=8):
        super().__init__()
        self.K, self.n_cells, self.d, self.d_u = K, n_cells, d, d_u
        self.cell_emb = nn.Parameter(torch.randn(n_cells, width) * 0.02)
        self.E = Gauss(width, 512, d); self.U = Gauss(n_cells * width, 512, d_u)
        self.z_in = nn.Linear(d, width); self.u_in = nn.Linear(d_u, width); self.c_in = nn.Linear(T5_DIM, width)
        self.f0_in = nn.Linear(DINO_DIM, width); self.pos = nn.Parameter(torch.randn(16 + n_cells + 2, width) * 0.02)
        enc = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.dec = nn.TransformerEncoder(enc, layers)
        self.out = nn.Linear(width, K * DINO_DIM)

    def encode(self, cells):
        zmu, zlv = self.E(cells + self.cell_emb); umu, ulv = self.U(cells.flatten(1))
        return zmu, zlv, umu, ulv

    def decode(self, f0, c, z, u):
        B = f0.shape[0]
        tok = torch.cat([self.f0_in(f0), self.z_in(z), self.u_in(u)[:, None], self.c_in(c)[:, None]], 1) + self.pos
        return self.out(self.dec(tok)[:, :16]).view(B, 16, self.K, DINO_DIM).transpose(1, 2)

    def encoder_parameters(self):
        return [self.cell_emb] + list(self.E.parameters()) + list(self.U.parameters())

    def freeze_encoder(self):
        for p in self.encoder_parameters(): p.requires_grad_(False)


# ---- losses ---------------------------------------------------------------------------------
def kl(mu, lv):
    return 0.5 * (mu.pow(2) + lv.exp() - 1 - lv).sum(-1).mean()


def reparam(mu, lv, train):
    return mu + torch.randn_like(mu) * (0.5 * lv).exp() if train else mu


def jl_projection(dim_in, dim_out=512, seed=0, dev="cpu"):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(dim_in, dim_out, generator=g) / math.sqrt(dim_out)).to(dev)


def mmd(x, y, proj):
    """RBF MMD^2 between two sample sets after a fixed JL projection; multi-bandwidth on the median."""
    x, y = x.flatten(1) @ proj, y.flatten(1) @ proj
    xy = torch.cat([x, y]); d2 = torch.cdist(xy, xy).pow(2)
    med = d2[d2 > 0].median().detach().clamp_min(1e-6)
    k = sum(torch.exp(-d2 / (med * s)) for s in (0.5, 1.0, 2.0))
    n = x.shape[0]; kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    return (kxx.sum() - kxx.diag().sum()) / (n * (n - 1)) + (kyy.sum() - kyy.diag().sum()) / (n * (n - 1)) - 2 * kxy.mean()


def derangement(n, dev, groups=None):
    """Permutation with no fixed point; if `groups` (episode ids) given, avoid same-episode partners."""
    perm = torch.randperm(n, device=dev)
    for _ in range(10):
        bad = perm == torch.arange(n, device=dev)
        if groups is not None: bad |= groups[perm] == groups
        if not bad.any(): break
        idx = bad.nonzero().flatten()
        perm[idx] = perm[idx[torch.randperm(len(idx), device=dev)]]
    return perm


# ---- checkpoints ----------------------------------------------------------------------------
def save_ckpt(path, model, args, dstats, extra=None):
    torch.save(dict(state_dict=model.state_dict(), args=dict(args), dstats=float(dstats), n_cells=model.n_cells, K=model.K,
                    d=model.d, d_u=model.d_u, **(getattr(model, "ckpt_meta", None) or {}), **(extra or {})), path)


def load_ckpt(path, dev):
    """Build the LAM described by a ckpt (this repo's or a legacy swaplam_*.pt) and load its weights."""
    ck = torch.load(path, map_location=dev)
    a = ck.get("args", {})
    if ck.get("kind") == "tcvq":                        # native-VQ task-centric tokenizer (common/tcvq.py)
        from .tcvq import TCVQ
        cfg = dict(ck["tcvq"]); K = cfg.pop("K"); m = TCVQ(K, **cfg).to(dev)
        m.load_state_dict(ck["state_dict"], strict=True); return m, float(ck["dstats"]), ck
    if ck.get("kind") == "pcrr":                        # portable core + residual on top of a frozen LAM (common/pcrr.py)
        from .pcrr import PCRR
        cfg = ck["pcrr"]; m = PCRR(LAM(**cfg["base"]).to(dev), cfg["d_p"], cfg["d_r"]).to(dev)
        m.load_state_dict(ck["state_dict"], strict=True); return m, float(ck["dstats"]), ck
    K = ck.get("K", a.get("T", 5) - 1); n_cells = ck.get("n_cells", 8)
    m = LAM(K, n_cells, ck.get("d", a.get("d", 32)), ck.get("d_u", a.get("d_u", 8)), layers=a.get("layers", 4)).to(dev)
    sd = {k: v for k, v in ck["state_dict"].items() if not k.startswith(("act_in.", "aux_head."))}   # legacy side heads
    missing, unexpected = m.load_state_dict(sd, strict=True), None
    return m, float(ck["dstats"]), ck


def legacy_path(tag):
    return os.path.join(LEGACY_CKPT, f"swaplam_{tag}.pt")
