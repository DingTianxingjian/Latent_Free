"""The continuous-latent action heads (ContHeads, FlowHeads), moved verbatim out of finetune_libero_ctrl.py on
2026-11-09 so that inference -- acdc_la/realrobot/piper_policy.py on a robot host -- can build them without importing
the training stack (prismatic's RLDS pipeline, FSDP, peft, ...).  finetune_libero_ctrl.py re-exports both names.

MAPBlock comes from acdc_la/vendor/univla/transformer_utils.py, a byte-identical copy of UniVLA's
prismatic/models/policy/transformer_utils.py, so the module tree and the state_dict keys are unchanged.
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F

from ...vendor.univla.transformer_utils import MAPBlock


class ContHeads(nn.Module):
    """Continuous-latent arm: no latent vocabulary.  The VLM's last hidden state is pooled into a prediction of the
    CONTINUOUS ACDC code, and the action head is conditioned on that code, so the only thing removed relative to the
    token arms is the quantiser.  Mirrors the oracle->predicted curriculum of downstream/libero_policy.py.

    LANGUAGE PATH (found 2026-09-18): the VLM input is [BOS, image, text] under a causal mask, so `visual` (image
    positions) never sees the instruction.  With text_direct=False the action head reads [visual summary, z] and the
    instruction reaches it ONLY through the regressed z (for dz = 0, not at all).  text_direct=True (`--head reg_text`)
    adds a pooled summary of the text positions to the action head's input and changes nothing else: the control that
    isolates that bottleneck from the regression-vs-flow change."""
    def __init__(self, dz, window_size=12, hidden=512, text_direct=False):
        super().__init__(); self.dz, self.text_direct = dz, text_direct
        self.vis_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden, n_heads=hidden // 64)
        self.z_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden, n_heads=hidden // 64)
        self.t_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden, n_heads=hidden // 64) if text_direct else None
        self.to_z = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, dz)) if dz else None
        self.act = nn.Sequential(nn.Linear(hidden * (2 if text_direct else 1) + dz, hidden), nn.GELU(), nn.Linear(hidden, 7 * window_size), nn.Tanh())

    def forward(self, visual, text, z_for_head=None):
        v = self.vis_pool(visual)
        u = [v, self.t_pool(text, init_embed=v)] if self.text_direct else [v]
        if self.dz == 0: return self.act(torch.cat(u, -1)), None     # --latent none: the same head with no latent input
        z_pred = self.to_z(self.z_pool(text, init_embed=v))
        z = z_pred if z_for_head is None else z_for_head
        return self.act(torch.cat(u + [z], -1)), z_pred


class FlowHeads(nn.Module):
    """Continuous arm with a FLOW-MATCHING head (2026-09-18): the action chunk and the continuous latent are generated
    jointly from the VLM's visual and prompt states, as RotVLA's head "jointly denoises latent and robot actions".

    Why it replaces ContHeads: ContHeads regresses the chunk with an L1/MSE target, the behaviour-cloning setting LAFP
    names as the cause of the policy-side failure ("collapse inherently multimodal action distributions into unimodal
    ones").  Generating z and the actions together also avoids LAFP's other named pitfall, a separately trained action
    decoder misaligned with a stochastic latent policy.  Rectified flow: x_t = (1-t) x0 + t x1, x0 ~ N(0, I), target
    velocity x1 - x0; sampling is `steps` Euler steps from noise.  Conditioning is the same MAP pooling of visual and
    prompt states that ContHeads and UniVLA's ActionDecoder use; the denoiser is ~12M parameters, the size of theirs.
    dz == 0 is the no-latent control through the identical head."""
    def __init__(self, dz, window_size=12, hidden=512, width=512, layers=4, z_tok=32):
        super().__init__(); self.dz, self.W, self.z_tok = dz, window_size, z_tok
        assert dz % z_tok == 0, (dz, z_tok)
        self.nz = dz // z_tok
        self.vis_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden, n_heads=hidden // 64)
        self.txt_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden, n_heads=hidden // 64)
        self.c_in = nn.Linear(2 * hidden, width)
        self.t_mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.a_in, self.a_out = nn.Linear(7, width), nn.Linear(width, 7)
        self.z_in, self.z_out = (nn.Linear(z_tok, width), nn.Linear(width, z_tok)) if dz else (None, None)
        self.pos = nn.Parameter(torch.randn(1, 1 + window_size + self.nz, width) * 0.02)
        self.blocks = nn.TransformerEncoder(nn.TransformerEncoderLayer(width, width // 64, 4 * width, 0.0, batch_first=True,
                                                                       norm_first=True), layers)
        self.norm = nn.LayerNorm(width)

    def cond(self, visual, text):
        v = self.vis_pool(visual)
        return self.c_in(torch.cat([v, self.txt_pool(text, init_embed=v)], -1))

    def _temb(self, t, width):
        f = torch.exp(-math.log(10000.0) * torch.arange(width // 2, device=t.device) / (width // 2))
        a = 1000.0 * t[:, None] * f[None]
        return self.t_mlp(torch.cat([a.sin(), a.cos()], -1))

    def velocity(self, xa, xz, t, c):
        h = [c[:, None], self.a_in(xa)] + ([self.z_in(xz)] if self.dz else [])
        h = torch.cat(h, 1) + self.pos + self._temb(t, c.shape[-1])[:, None]
        h = self.norm(self.blocks(h))
        return self.a_out(h[:, 1:1 + self.W]), (self.z_out(h[:, 1 + self.W:]) if self.dz else None)

    def loss(self, visual, text, actions, z_star=None):
        """-> (action flow loss, latent flow loss, one-step action L1, one-step latent R2); the last two are diagnostics."""
        c = self.cond(visual, text); B = actions.shape[0]
        t = torch.rand(B, device=actions.device)
        a0 = torch.randn_like(actions); at = (1 - t[:, None, None]) * a0 + t[:, None, None] * actions
        z1 = z_star.reshape(B, self.nz, self.z_tok) if self.dz else None
        z0 = torch.randn_like(z1) if self.dz else None
        zt = (1 - t[:, None, None]) * z0 + t[:, None, None] * z1 if self.dz else None
        va, vz = self.velocity(at, zt, t, c)
        la = F.mse_loss(va, actions - a0)
        with torch.no_grad():
            l1 = (at + (1 - t[:, None, None]) * va - actions).abs().mean()           # x1 estimated in one step
        if not self.dz:
            z0_ = torch.zeros((), device=actions.device); return la, z0_, l1, z0_
        lz = F.mse_loss(vz, z1 - z0)
        with torch.no_grad():
            zh = (zt + (1 - t[:, None, None]) * vz).reshape(B, -1)
            r2 = 1 - F.mse_loss(zh, z_star) / z_star.var(0).mean().clamp_min(1e-6)
        return la, lz, l1, r2

    @torch.no_grad()
    def sample(self, visual, text, steps=10, generator=None):
        c = self.cond(visual, text); B, dev = c.shape[0], c.device
        xa = torch.randn(B, self.W, 7, device=dev, generator=generator)
        xz = torch.randn(B, self.nz, self.z_tok, device=dev, generator=generator) if self.dz else None
        for k in range(steps):
            t = torch.full((B,), k / steps, device=dev)
            va, vz = self.velocity(xa, xz, t, c)
            xa = xa + va / steps
            if self.dz: xz = xz + vz / steps
        return xa.clamp(-1, 1), (xz.reshape(B, -1) if self.dz else None)
