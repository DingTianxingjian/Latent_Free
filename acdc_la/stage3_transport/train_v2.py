"""LATP v2 -- interventional transport with an EMA target encoder and Relational Transport Consistency.

Why v2 exists.  Stage III freezes the tokenizer, so it trains only T.  The VLA labeller uses ONLY the encoder
(`downstream/vla/tcvq_tokens.py` calls `pre_quant`/`quantize`, never `decode`), therefore nothing Stage III learns
can reach the policy.  v2 makes the encoder trainable and gives it supervision that a reconstruction loss cannot
provide: the latent identity of a SELF-CHOSEN intervention, which is synthetic supervision that needs no
counterfactual ground truth (the CycleMimic principle).

    L = L_self  +  lam_recover * L_recover  +  lam_real * L_MMD  +  lam_rel * L_RTC

  L_self     real clip:      z_i = E(F_i0, dF_i),  || T(F_i0, z_i) - dF_i ||^2          keeps E and T on real data
  L_recover  intervention:   z_s ~ buffer of EMA-TARGET latents (same source, other episode),
                             Y = T(F_j0, z_s),  z^ = E(F_j0, Y),   1 - cos(z~^, sg z~) + l1 * SmoothL1
  L_MMD      realism:        MMD({Y}, {dF_real}) -- without it T can watermark z_s into Y (see WATERMARK below)
  L_RTC      relational:     with Psi(Y) = whitened random projection of [sum_k Y_k, sum_k b_1(k) Y_k],
                             r_j^ab = Psi(T(s_j,z_a)) - Psi(T(s_j,z_b)),   1 - cos(r_j^ab, r_k^ab)
                             "the relation between two interventions realises in a context-stable direction"

Gradient routing.  E: L_self + L_recover.  T: L_self + L_MMD + L_RTC (+ L_recover unless --recover-detach).
Target encoder: EMA only, no gradient.  RTC touches T alone -- its latents come from the stop-grad buffer -- so a
positive RTC result localises to the operator, as intended.

WATERMARK.  L_recover is also minimised by T encoding z_s into Y in any way E can read, with no action semantics.
MMD on the marginal of Y is a blunt guard against a low-amplitude code.  Two responses are built in:
  * --recover-detach cuts the channel entirely (L_recover then trains E only, which is the stated purpose);
  * `rec_noise` logs recovery cosine after Gaussian noise at --probe-noise x dF rms is added to Y.  A watermark is
    low-amplitude and dies under that noise; real effect structure does not.  Watch rec_cos - rec_noise.

RTC DEGENERACY.  L_RTC = 0 for any T whose effect descriptor is additively separable, Psi(T(s,z)) = h(s) + g(z),
because the difference r cancels h(s).  Its global minimum therefore includes operators that under-use the
context.  `ctx_sens` logs the relative distance between Psi(T(s_j,z_a)) and Psi(T(s_k,z_a)); if it falls toward 0
the RTC term is being satisfied by ignoring the state and the run is not evidence for anything.

    python -m acdc_la.stage3_transport.train_v2 --tag latp2 --init-from tc_cont --steps 30000 --batch 64
    python -m acdc_la.stage3_transport.train_v2 --tag latp2_norel --init-from tc_cont --lam-rel 0     # ablation
    python -m acdc_la.stage3_transport.train_v2 --tag latp2_frozen --init-from tc_cont --freeze-encoder  # = Stage III
"""
import argparse, copy
import torch
import torch.nn.functional as F

from ..common.train_loop import common_args, run
from ..common.model import mmd, derangement, jl_projection
from ..common import features as Fz


# ---- state carried across steps (built lazily on the first step; run() owns the loop) -------------------
def _ctx(model, b, a, ctx):
    if "teacher" in ctx: return ctx
    dev = ctx["dev"]
    ctx["teacher"] = copy.deepcopy(model).eval().requires_grad_(False)
    ctx["buf"] = []                                                    # [(z [n,d], si scalar, ge scalar)] x M
    ctx["zmu"] = torch.zeros(model.d, device=dev); ctx["zvar"] = torch.ones(model.d, device=dev)
    P = b["dF"].shape[2] * b["dF"].shape[3]
    ctx["psi_proj"] = jl_projection(2 * P, a.psi_dim, 1, dev)          # FIXED: Psi takes no gradient parameters
    ctx["psi_mu"] = torch.zeros(a.psi_dim, device=dev); ctx["psi_var"] = torch.ones(a.psi_dim, device=dev)
    ctx["b1"] = Fz.dct_basis(model.K, 2, dev)[1]
    return ctx


def _ema_(dst, src, tau):
    with torch.no_grad():
        for pd, ps in zip(dst.parameters(), src.parameters()): pd.mul_(tau).add_(ps.detach(), alpha=1 - tau)
        for bd, bs in zip(dst.buffers(), src.buffers()): bd.copy_(bs)


def _running_(mu, var, x, m=0.99):
    with torch.no_grad():
        mu.mul_(m).add_(x.mean(0), alpha=1 - m); var.mul_(m).add_(x.var(0, unbiased=False), alpha=1 - m)


def _std(z, mu, var):
    return (z - mu) / (var.sqrt() + 1e-5)


def _code(model, f0, dF):
    return model.quantize(f0, dF)[0]


def _psi(Y, ctx):
    """Fixed effect descriptor: [net, AC1] of a transition -> fixed random projection -> running whitening.
    No trainable parameter, but the gradient reaches T through these linear operations."""
    n = Y.sum(1).flatten(1); ac = torch.einsum("k,bkpd->bpd", ctx["b1"].to(Y.dtype), Y).flatten(1)
    p = torch.cat([n, ac], 1) @ ctx["psi_proj"]
    return (p - ctx["psi_mu"]) / (ctx["psi_var"].sqrt() + 1e-5), p


def _donors(ctx, si, ge, n_tokens, d, dev):
    """One buffer latent per row: same source, different episode.  Returns (z [B,n,d], ok mask [B])."""
    buf = ctx["buf"]
    if not buf: return torch.zeros(len(si), n_tokens, d, device=dev), torch.zeros(len(si), dtype=torch.bool, device=dev)
    Z = torch.stack([e[0] for e in buf]); S = torch.tensor([e[1] for e in buf]); G = torch.tensor([e[2] for e in buf])
    out = torch.zeros(len(si), n_tokens, d, device=dev); ok = torch.zeros(len(si), dtype=torch.bool)
    for j in range(len(si)):
        cand = ((S == si[j]) & (G != ge[j])).nonzero().flatten()
        if len(cand): out[j] = Z[cand[torch.randint(len(cand), (1,))].item()]; ok[j] = True
    return out, ok.to(dev)


def _rtc_groups(ctx, si, ge, n_max):
    """Groups (j, k, a, b): two contexts from one source in this batch, two donors from that source's buffer."""
    buf = ctx["buf"]
    if not buf: return []
    S = torch.tensor([e[1] for e in buf]); g = []
    by_src = {}
    for i, s in enumerate(si.tolist()): by_src.setdefault(s, []).append(i)
    for s, rows in by_src.items():
        if len(rows) < 2 or len(g) >= n_max: continue
        cand = (S == s).nonzero().flatten()
        if len(cand) < 2: continue
        j, k = rows[torch.randint(len(rows), (1,)).item()], rows[torch.randint(len(rows), (1,)).item()]
        if ge[j] == ge[k]: continue                                    # different episodes = genuinely different contexts
        p = cand[torch.randperm(len(cand))[:2]]
        g.append((j, k, int(p[0]), int(p[1])))
    return g


def step(model, feat, b, a, ctx, train=True):
    fresh = "teacher" not in ctx
    ctx = _ctx(model, b, a, ctx); dev = ctx["dev"]; B = b["f0"].shape[0]
    if not fresh: _ema_(ctx["teacher"], model, a.tau)     # after the previous optimiser step, before this forward
    si, ge = b["si"].cpu(), b["ge"].cpu()
    out = {}

    # ---- L_self: the online tokenizer on real clips ---------------------------------------------------
    z = _code(model, b["f0"], b["dF"])
    out["rec"] = F.mse_loss(model.decode(b["f0"], b["c"], z), b["dF"])
    loss = out["rec"]
    _running_(ctx["zmu"], ctx["zvar"], z.detach().flatten(0, 1))

    # ---- the EMA target's view of the same clips goes into the donor buffer ----------------------------
    with torch.no_grad():
        zt = _code(ctx["teacher"], b["f0"], b["dF"])
    for i in range(B): ctx["buf"].append((zt[i], int(si[i]), int(ge[i])))
    del ctx["buf"][:max(0, len(ctx["buf"]) - a.buffer * B)]

    # ---- L_recover: a self-chosen intervention must be readable back out of its realisation ------------
    zs, ok = _donors(ctx, si, ge, model.n_tokens, model.d, dev)
    if ok.any():
        Y = model.decode(b["f0"], b["c"], zs)                              # donor code on THIS row's state
        Yr = Y.detach() if a.recover_detach else Y
        zh = _code(model, b["f0"], Yr)
        zt_s, zh_s = _std(zs, ctx["zmu"], ctx["zvar"]), _std(zh, ctx["zmu"], ctx["zvar"])
        cos = F.cosine_similarity(zh_s.flatten(1), zt_s.flatten(1), dim=1)
        sl1 = F.smooth_l1_loss(zh_s, zt_s, reduction="none").flatten(1).mean(1)
        out["rec_cos"] = cos[ok].mean()
        out["recover"] = ((1 - cos) + a.recover_l1 * sl1)[ok].mean()
        out["mmd"] = mmd(Y[ok], b["dF"][ok], ctx["proj"])
        loss = loss + a.lam_recover * out["recover"] + a.lam_real * out["mmd"]
        with torch.no_grad():                                              # WATERMARK probe: does recovery survive noise?
            Yn = Y + torch.randn_like(Y) * a.probe_noise * Y.detach().flatten(1).norm(dim=1).mean() / (Y[0].numel() ** 0.5)
            zn = _std(_code(model, b["f0"], Yn), ctx["zmu"], ctx["zvar"])
            out["rec_noise"] = F.cosine_similarity(zn.flatten(1), zt_s.flatten(1), dim=1)[ok].mean()
        out["donor_frac"] = ok.float().mean()

    # ---- L_RTC: the relation between two interventions realises in a context-stable direction ----------
    G = _rtc_groups(ctx, si, ge, a.rtc_groups) if a.lam_rel > 0 else []
    if G:
        with torch.no_grad(): _running_(ctx["psi_mu"], ctx["psi_var"], _psi(b["dF"], ctx)[1])   # whitening from REAL transitions
        Zb = torch.stack([e[0] for e in ctx["buf"]])
        jj = torch.tensor([g[0] for g in G]); kk = torch.tensor([g[1] for g in G])
        za, zb = Zb[[g[2] for g in G]], Zb[[g[3] for g in G]]
        f0 = torch.cat([b["f0"][jj], b["f0"][jj], b["f0"][kk], b["f0"][kk]])
        c = torch.cat([b["c"][jj], b["c"][jj], b["c"][kk], b["c"][kk]])
        Yall = model.decode(f0, c, torch.cat([za, zb, za, zb]))
        psi, _ = _psi(Yall, ctx); n = len(G)
        pja, pjb, pka, pkb = psi[:n], psi[n:2 * n], psi[2 * n:3 * n], psi[3 * n:]
        out["rtc_cos"] = F.cosine_similarity(pja - pjb, pka - pkb, dim=1).mean()
        out["rtc"] = 1 - out["rtc_cos"]
        loss = loss + a.lam_rel * out["rtc"]
        with torch.no_grad():                                              # DEGENERACY probe: is T still using the state?
            out["ctx_sens"] = ((pja - pka).norm(dim=1) / pja.norm(dim=1).clamp_min(1e-6)).mean()
    out["loss"] = loss
    return loss, out


def main():
    ap = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--lam-recover", type=float, default=1.0); ap.add_argument("--lam-real", type=float, default=1.0)
    ap.add_argument("--lam-rel", type=float, default=0.5); ap.add_argument("--recover-l1", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.99, help="EMA rate of the target encoder (SPR/BYOL range)")
    ap.add_argument("--buffer", type=int, default=16, help="donor buffer depth, in batches of TARGET latents")
    ap.add_argument("--rtc-groups", type=int, default=16, help="(context, context, donor, donor) groups per step; costs 4x decodes")
    ap.add_argument("--psi-dim", type=int, default=512)
    ap.add_argument("--probe-noise", type=float, default=0.1, help="noise level of the watermark probe, relative to the transition rms")
    ap.add_argument("--recover-detach", action="store_true",
                    help="cut T out of L_recover: removes the watermark channel entirely and leaves L_recover training E only")
    a = ap.parse_args()
    assert a.init_from or a.eval_only, "LATP v2 starts from a trained tokenizer (--init-from)"
    run(a, step, ramp_keys=("lam_recover", "lam_real", "lam_rel"), stage="LATP2")

if __name__ == "__main__":
    main()
