"""Structured temporal tokenizer with a NATIVE discrete bottleneck, at UniVLA's exact token budget (review 2026-09-14).

The downstream interface is fixed to theirs -- the policy predicts `n_tokens` indices of one `codebook`-entry codebook --
so the arms differ only in WHAT EVIDENCE the encoder sees about the chunk, at patch level, through the same network:

    state  S_p = F_{0,p}                      (4x4 grid, 16 tokens)   absolute configuration
    net    N_p = sum_k dF_{k,p} = F_{T,p} - F_{0,p}                    the ENDPOINT displacement: (S, N) is an exact
                                                                       reparameterisation of (F_0, F_T), so the encoder
                                                                       is given endpoint evidence without being forced
                                                                       to consume it as a subtraction
    dyn    A_p = sum_k b_1(k) dF_{k,p}                                 the intra-chunk temporal mode, orthogonal to N
                                                                       and NOT recoverable from the endpoints

    tokens -> (linear + spatial pos + type embedding) -> Transformer with `n_tokens` latent queries
           -> VQ (UniVLA's own: straight-through, dead-code restart) -> z_q -> D(F_0, c, z_q) -> dF^

Arms (identical network, bitrate, data, optimiser; only the token set changes):
    endpoint     [S, N]        the evidence UniVLA's labeller has (it encodes the pair (F_0, F_T))
    endpointdyn  [S, N, A]     the method: one extra statistic the endpoints cannot express
    dyn          [S, A]        sanity arm: must be worse on net displacement
Reconstruction is computed from the QUANTISED code, so the tokenizer optimises exactly what the VLA receives.
"""
import sys, math
import torch, torch.nn as nn

from ..paths import DINO_DIM, T5_DIM, UNIVLA_ROOT
from .model import LAM
from . import features as Fz

sys.path.insert(0, UNIVLA_ROOT)
from latent_action_model.genie.modules.blocks import VectorQuantizer      # noqa: E402  matched bottleneck

ARMS = {"endpoint": ("S", "N"), "endpointdyn": ("S", "N", "A"), "dyn": ("S", "A"), "netonly": ("N",)}


class TCVQ(nn.Module):
    swap_u_zero = True                      # no second channel: the dump's swap path must not feed one

    def __init__(self, K, tc="endpointdyn", n_tokens=4, codebook=16, latent_dim=128, width=768, enc_layers=6,
                 heads=12, dec_layers=4, grid=16, bottleneck="vq"):
        super().__init__()
        assert tc in ARMS, tc; assert bottleneck in ("vq", "continuous"), bottleneck
        self.bottleneck = bottleneck
        self.K, self.tc, self.n_tokens, self.codebook_size, self.latent_dim, self.grid = K, tc, n_tokens, codebook, latent_dim, grid
        self.kinds = ARMS[tc]
        self.inp = nn.ModuleDict({k: nn.Sequential(nn.LayerNorm(DINO_DIM), nn.Linear(DINO_DIM, width)) for k in self.kinds})
        self.type_emb = nn.Parameter(torch.randn(len(self.kinds), width) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(grid, width) * 0.02)
        self.query = nn.Parameter(torch.randn(n_tokens, width) * 0.02)
        enc = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, enc_layers)
        self.to_code = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, latent_dim))
        # DEPLOYMENT PRIOR (review 2026-09-15).  p(z | F_0, language) sees only what the policy will see at deployment;
        # pulling the posterior towards it prices content that the deployment information set cannot determine.  The
        # posterior encoder is unchanged, so the downstream labeller needs no language at all.
        self.prior_lang = nn.Sequential(nn.LayerNorm(T5_DIM), nn.Linear(T5_DIM, width))
        self.prior_query = nn.Parameter(torch.randn(n_tokens, width) * 0.02)
        pe = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.prior_enc = nn.TransformerEncoder(pe, 2)
        self.prior_to_code = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, latent_dim))
        self.vq = VectorQuantizer(codebook, latent_dim) if bottleneck == "vq" else None
        self.dec = LAM(K, n_cells=n_tokens, d=latent_dim, d_u=8, layers=dec_layers)    # decoder only; its E/U are unused
        self.n_cells, self.d, self.d_u = n_tokens, latent_dim, 8                       # train_loop / dump compatibility
        self.register_buffer("b1", Fz.dct_basis(K, 2)[1], persistent=False)            # orthonormal AC1 over the K segments
        self.ckpt_meta = dict(kind="tcvq", tcvq=dict(tc=tc, n_tokens=n_tokens, codebook=codebook, latent_dim=latent_dim,
                                                     width=width, enc_layers=enc_layers, heads=heads,
                                                     dec_layers=dec_layers, grid=grid, K=K, bottleneck=bottleneck))

    def evidence(self, f0, dF):
        """f0 (B,16,768) first-frame g4, dF (B,K,16,768) scale-normalised segment deltas -> the token set of this arm."""
        out = {}
        if "S" in self.kinds: out["S"] = f0
        if "N" in self.kinds: out["N"] = dF.sum(1)
        if "A" in self.kinds: out["A"] = torch.einsum("k,bkpd->bpd", self.b1.to(dF.dtype), dF)
        return out

    def pre_quant(self, f0, dF):
        ev = self.evidence(f0, dF); B = f0.shape[0]
        toks = [self.inp[k](ev[k]) + self.pos_emb[None] + self.type_emb[i][None, None] for i, k in enumerate(self.kinds)]
        x = torch.cat([self.query[None].expand(B, -1, -1)] + toks, 1)
        return self.to_code(self.enc(x)[:, :self.n_tokens])

    def prior_code(self, f0, c):
        """p(z | F_0, language): the deployment information set only -- no future frames."""
        B = f0.shape[0]
        toks = self.inp[self.kinds[0]](f0) + self.pos_emb[None]          # the state tokens, same projection as "S"
        x = torch.cat([self.prior_query[None].expand(B, -1, -1), toks, self.prior_lang(c)[:, None]], 1)
        return self.prior_to_code(self.prior_enc(x)[:, :self.n_tokens])

    def quantize(self, f0, dF):
        """vq: (z_q straight-through, z codebook embeddings, x pre-quantisation, indices (B, n_tokens)).
        continuous: the code itself in all three slots and an empty index tensor -- no quantiser, no rate cap."""
        x = self.pre_quant(f0, dF)
        if self.vq is None: return x, x, x, torch.zeros(x.shape[0], 0, dtype=torch.long, device=x.device)
        return self.vq(x)

    def _b(self, b, which):
        f0 = b["f0_patch" if which == "patch" else "f0"]; dF = b["dF_patch" if which == "patch" else "dF"]
        return f0, (-dF.flip(1) if which == "rev" else dF)

    def encode_batch(self, b, which="clean"):
        """LAM-compatible 4-tuple so common/dump.py and every probe read this model unchanged:
        (z_mu = the quantised code, z_lv = 0, u_mu = 0, u_lv = 0).  There is no second channel and no KL."""
        z_q = self.quantize(*self._b(b, which))[0]
        zero_u = torch.zeros(z_q.shape[0], self.d_u, device=z_q.device, dtype=z_q.dtype)
        return z_q, torch.zeros_like(z_q), zero_u, zero_u

    def indices_of(self, f0, dF):
        return self.quantize(f0, dF)[3]

    def decode(self, f0, c, z, u=None):
        u = torch.zeros(f0.shape[0], self.d_u, device=f0.device, dtype=f0.dtype) if u is None else u
        return self.dec.decode(f0, c, z, u)

    def encoder_parameters(self):
        return [self.type_emb, self.pos_emb, self.query] + list(self.inp.parameters()) + list(self.enc.parameters()) + list(self.to_code.parameters())

    def freeze_encoder(self):
        for p in self.encoder_parameters(): p.requires_grad_(False)
