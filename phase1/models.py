"""Phase-1 models.  Model A = LAM baseline (UniVLA-style).  PRLA added in stage 2.

All operate on frozen V-JEPA2 g4 tokens: a window is (NFRAMES, 16, 1024).

LAM (self-supervised, no actions in the loss):
  IDM   spatio-temporal transformer over the window's 5x16 tokens + n_q query
        tokens -> n_q continuous codes `emb`.
  VQ    n_q independent codebooks (size 16) -> quantized `z_q` (+ indices).
  FDM   spatial transformer over CLEAN f(o_t)'s 16 tokens + the n_q codes ONLY,
        predicting f(o_{t+span})'s 16 tokens.  Feature-space MSE.

Critical constraint (UniVLA's whole point): the ONLY path from the future frames
to the FDM is the quantized codes.  Enforced structurally — FDM.forward takes
(ot_tokens, z_q) and literally has no argument for window/IDM tokens; asserted in
the smoke test.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _encoder(d, layers, heads=8, ff_mult=4):
    layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=ff_mult * d,
                                       dropout=0.0, batch_first=True, activation="gelu",
                                       norm_first=True)
    return nn.TransformerEncoder(layer, layers)


class IDM(nn.Module):
    """Window tokens (+ query tokens) -> n_q continuous codes."""

    def __init__(self, nframes, ntok, hid, d=512, layers=6, n_q=4, d_code=64):
        super().__init__()
        self.nframes, self.ntok, self.n_q = nframes, ntok, n_q
        self.inp = nn.Linear(hid, d)
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, ntok, d))
        self.temporal_pos = nn.Parameter(torch.zeros(1, nframes, 1, d))
        self.query = nn.Parameter(torch.zeros(1, n_q, d))
        nn.init.normal_(self.spatial_pos, std=0.02)
        nn.init.normal_(self.temporal_pos, std=0.02)
        nn.init.normal_(self.query, std=0.02)
        self.enc = _encoder(d, layers)
        self.out = nn.Linear(d, d_code)

    def forward(self, win):                             # win: (B, nframes, ntok, hid)
        B = win.shape[0]
        x = self.inp(win) + self.spatial_pos + self.temporal_pos   # (B,F,T,d)
        x = x.reshape(B, self.nframes * self.ntok, -1)             # (B, F*T, d)
        q = self.query.expand(B, -1, -1)                           # (B, n_q, d)
        h = self.enc(torch.cat([x, q], dim=1))                    # (B, F*T+n_q, d)
        return self.out(h[:, -self.n_q:])                          # (B, n_q, d_code) = emb


class VQ(nn.Module):
    """n_q independent codebooks of size K.  Straight-through, commitment loss."""

    def __init__(self, n_q=4, K=16, d_code=64, commit=0.25):
        super().__init__()
        self.n_q, self.K, self.commit = n_q, K, commit
        self.codebook = nn.Parameter(torch.randn(n_q, K, d_code) * 0.1)

    def forward(self, emb):                              # emb: (B, n_q, d_code)
        B = emb.shape[0]
        # per-position nearest code (each of the n_q positions has its own codebook)
        idx = torch.empty(B, self.n_q, dtype=torch.long, device=emb.device)
        zq = torch.empty_like(emb)
        for p in range(self.n_q):
            dist = torch.cdist(emb[:, p], self.codebook[p])          # (B, K)
            ip = dist.argmin(1)                                      # (B,)
            idx[:, p] = ip
            zq[:, p] = self.codebook[p][ip]
        codebook_loss = F.mse_loss(zq, emb.detach())
        commit_loss = F.mse_loss(emb, zq.detach())
        vq_loss = codebook_loss + self.commit * commit_loss
        zq_st = emb + (zq - emb).detach()                # straight-through
        return zq_st, idx, vq_loss


class FDM(nn.Module):
    """CLEAN f(o_t) tokens + codes -> predict f(o_{t+span}) tokens.

    forward(ot_tokens, z_q) — NO window/IDM argument exists, by design."""

    def __init__(self, ntok, hid, d=512, layers=4, n_q=4, d_code=64):
        super().__init__()
        self.ntok = ntok
        self.ot_inp = nn.Linear(hid, d)
        self.code_inp = nn.Linear(d_code, d)
        self.ot_pos = nn.Parameter(torch.zeros(1, ntok, d))
        self.code_pos = nn.Parameter(torch.zeros(1, n_q, d))
        nn.init.normal_(self.ot_pos, std=0.02)
        nn.init.normal_(self.code_pos, std=0.02)
        self.enc = _encoder(d, layers)
        self.out = nn.Linear(d, hid)

    def forward(self, ot_tokens, z_q):                   # (B,ntok,hid), (B,n_q,d_code)
        o = self.ot_inp(ot_tokens) + self.ot_pos
        c = self.code_inp(z_q) + self.code_pos
        h = self.enc(torch.cat([o, c], dim=1))
        return self.out(h[:, :self.ntok])                # (B, ntok, hid) predicted end tokens


class LAM(nn.Module):
    def __init__(self, nframes, ntok, hid, d=512, idm_layers=6, fdm_layers=4,
                 n_q=4, K=16, d_code=64, commit=0.25):
        super().__init__()
        self.idm = IDM(nframes, ntok, hid, d, idm_layers, n_q, d_code)
        self.vq = VQ(n_q, K, d_code, commit)
        self.fdm = FDM(ntok, hid, d, fdm_layers, n_q, d_code)
        self.n_q, self.K, self.d_code = n_q, K, d_code

    def encode(self, win):
        """-> emb (B,n_q,d_code), z_q (B,n_q,d_code), idx (B,n_q), vq_loss."""
        emb = self.idm(win)
        zq, idx, vq_loss = self.vq(emb)
        return emb, zq, idx, vq_loss

    def forward(self, win, ot_tokens, target_tokens):
        emb, zq, idx, vq_loss = self.encode(win)
        pred = self.fdm(ot_tokens, zq)
        recon = F.mse_loss(pred, target_tokens)
        loss = recon + vq_loss
        return dict(loss=loss, recon=recon, vq_loss=vq_loss, idx=idx, emb=emb, zq=zq)


# ======================= Model B — PRLA (the method) ======================= #
# Predictive-residual latent action.  z = what a frozen passive predictor cannot
# explain.  Operates on FUSED window latents (V-JEPA2 clip -> g4 tmean, 16384-d).
# No VQ: continuous z + KL, our own measurements justify it.

def _mlp(din, dh, dout, layers=2):
    mods, d = [], din
    for _ in range(layers):
        mods += [nn.Linear(d, dh), nn.GELU()]
        d = dh
    mods += [nn.Linear(d, dout)]
    return nn.Sequential(*mods)


class PassivePredictor(nn.Module):
    """P_A: history fused latent -> prediction of future fused latent.  Trained
    to convergence and FROZEN before z exists, so it has no incentive to under-fit
    and leave predictable content for z to carry."""

    def __init__(self, dim, hidden=1024, layers=3):
        super().__init__()
        self.net = _mlp(dim, hidden, dim, layers)

    def forward(self, hist):
        return self.net(hist)


class ResidualEncoder(nn.Module):
    """E: sees (true future, P_A's prediction) -> (mu, logvar) of a low-dim z."""

    def __init__(self, dim, d_z, hidden=1024, layers=2):
        super().__init__()
        self.net = _mlp(2 * dim, hidden, 2 * d_z, layers)
        self.d_z = d_z

    def forward(self, fut, pred_a):
        h = self.net(torch.cat([fut, pred_a], dim=-1))
        mu, logvar = h[:, :self.d_z], h[:, self.d_z:]
        return mu, logvar


class Corrector(nn.Module):
    """G: correction only.  pred_B = pred_A + G(z, pred_A) — residual hard-coded."""

    def __init__(self, dim, d_z, hidden=1024, layers=2):
        super().__init__()
        self.net = _mlp(d_z + dim, hidden, dim, layers)

    def forward(self, z, pred_a):
        return pred_a + self.net(torch.cat([z, pred_a], dim=-1))


class PRLA(nn.Module):
    def __init__(self, dim, d_z=16, hidden=1024):
        super().__init__()
        self.p_a = PassivePredictor(dim, hidden)
        self.enc = ResidualEncoder(dim, d_z, hidden)
        self.gen = Corrector(dim, d_z, hidden)
        self.dim, self.d_z = dim, d_z

    def predict_a(self, hist):
        return self.p_a(hist)

    def encode(self, fut, pred_a):
        mu, logvar = self.enc(fut, pred_a)
        return mu, logvar

    @staticmethod
    def reparam(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, hist, fut, beta):
        """Stage-2 loss with P_A frozen (caller freezes p_a's grads)."""
        pred_a = self.p_a(hist).detach()             # P_A is frozen; no grad through it
        mu, logvar = self.enc(fut, pred_a)
        z = self.reparam(mu, logvar)
        pred_b = self.gen(z, pred_a)
        recon = F.mse_loss(pred_b, fut)
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean()   # nats
        loss = recon + beta * kl
        # correction gain vs the passive baseline (for the rate-distortion curve)
        with torch.no_grad():
            base = F.mse_loss(pred_a, fut)
        return dict(loss=loss, recon=recon, kl=kl, base_mse=base, mu=mu, logvar=logvar)
