"""History-conditioned, chain-trained latent policy for LIBERO-spatial (review 2026-09-12): the downstream recipe.

    z^_t  = pi(f_{t-3..t}, task)                 small causal transformer over 4 frames of frozen DINO g4 (quadrant-pooled)
    a^    = H_BC(f_t, task) + H_lat(f_t, z_mix)   residual BC path; H_lat's last layer is zero-initialised
    z_mix = z* with prob p (annealed 1 -> 0) else z^   oracle -> predicted curriculum so H trains on the deployed distribution
    L = Huber(a^, a) + lam_z ||z^ - z*||^2 + lam_e ||T(s_t, z^) - sg T(s_t, z*)||^2     (T = frozen LATP decoder, u = 0, c = 0)

Arms (same architecture, demos, optimiser, chunk, rollout protocol):
    bc          H_BC only
    ours        z* from --tokenizer (ACDC), T from --transport (LATP), all three losses
    ours_noeff  same without the effect loss (lam_e = 0)
    univla      z* = budget-matched UniVLA codes (--univla-arm), latent loss only (their decoder is not T)
Offline: L1 / token accuracy on held-out episodes for oracle and predicted latents.  Closed-loop: --rollout N starts per task.
"""
import os, sys, json, time, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from ..paths import RESULTS, CKPT, IMG
from ..common import features as Fz
from ..common.model import load_ckpt, legacy_path
from . import libero_offline as L
from .libero_offline import libero, K, S, N_TASKS

HIST = 4


def hist_index(T_, E_, idx):
    """(n, HIST) frame ids t-3..t clamped at the episode start."""
    starts = idx["starts"][E_]
    return np.stack([np.maximum(T_ - h, starts) for h in range(HIST - 1, -1, -1)], 1)


class Policy(nn.Module):
    def __init__(self, dz, width=256, layers=2):
        super().__init__()
        self.inp = nn.Linear(768, width); self.task = nn.Embedding(N_TASKS, width)
        self.pos = nn.Parameter(torch.randn(1, HIST * 4 + 1, width) * 0.02)
        enc = nn.TransformerEncoderLayer(width, 4, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers); self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dz))

    def forward(self, fq, task):                   # fq (B,HIST,4,768) quadrant-pooled history, task (B,) long
        B = fq.shape[0]
        tok = torch.cat([self.task(task)[:, None], self.inp(fq.reshape(B, HIST * 4, 768))], 1) + self.pos
        return self.out(self.tr(tok)[:, 0])


def mlp(i, o, h=512, zero_last=False):
    m = nn.Sequential(nn.LayerNorm(i), nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))
    if zero_last: nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)
    return m


class Agent(nn.Module):
    def __init__(self, dz, use_latent):
        super().__init__(); self.use_latent = use_latent
        self.bc = mlp(3072 + N_TASKS, K * S * 7)
        if use_latent: self.pi = Policy(dz); self.lat = mlp(3072 + dz, K * S * 7, zero_last=True)

    def forward(self, fq_hist, task, z_for_head=None):
        onehot = F.one_hot(task, N_TASKS).float(); f_t = fq_hist[:, -1].reshape(len(task), -1)
        a = self.bc(torch.cat([f_t, onehot], 1)); zhat = None
        if self.use_latent:
            zhat = self.pi(fq_hist, task); z_use = zhat if z_for_head is None else z_for_head
            a = a + self.lat(torch.cat([f_t, z_use], 1))
        return a, zhat


def train_agent(arm, feats, dev, a, T_model=None):
    """feats: dict with FQ (n,HIST,4,768) fp16 cpu, F0 (n,16,768) fp16 cpu (for T), task, chunk, Zstar (n,dz) or None; tr mask."""
    tr = feats["tr"]; n_tr = int(tr.sum()); idx_tr = np.flatnonzero(tr)
    dz = feats["Zstar"].shape[1] if feats["Zstar"] is not None else 0
    agent = Agent(dz, arm != "bc").to(dev)
    opt = torch.optim.AdamW(agent.parameters(), lr=3e-4, weight_decay=0.01); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    chunk_mu = torch.as_tensor(feats["chunk"][tr].mean(0), device=dev); chunk_sd = torch.as_tensor(feats["chunk"][tr].std(0) + 1e-6, device=dev)
    g = torch.Generator().manual_seed(0); t0 = time.time(); log = {}
    for it in range(a.iters):
        b = idx_tr[torch.randint(0, n_tr, (a.batch,), generator=g).numpy()]
        fq = feats["FQ"][b].to(dev).float(); task = torch.as_tensor(feats["task"][b], device=dev)
        act = (torch.as_tensor(feats["chunk"][b], device=dev) - chunk_mu) / chunk_sd
        zstar = torch.as_tensor(feats["Zstar"][b], device=dev) if dz else None
        p = max(0.0, 1.0 - it / (0.6 * a.iters))                                    # oracle -> predicted curriculum
        if arm == "bc":
            pred, _ = agent(fq, task); loss = F.huber_loss(pred, act); terms = dict(act=loss)
        else:
            zhat = agent.pi(fq, task)
            use_star = (torch.rand(len(b), 1, device=dev) < p).float()
            z_mix = use_star * zstar + (1 - use_star) * zhat
            pred, _ = agent(fq, task, z_for_head=z_mix)
            terms = dict(act=F.huber_loss(pred, act), lat=F.mse_loss(zhat, zstar))
            loss = terms["act"] + a.lam_z * terms["lat"]
            if T_model is not None and a.lam_e > 0:
                f0 = feats["F0"][b].to(dev).float(); u0 = torch.zeros(len(b), T_model.d_u, device=dev); c0 = torch.zeros(len(b), 768, device=dev)
                eff_hat = T_model.decode(f0, c0, zhat.view(len(b), T_model.n_cells, T_model.d), u0)
                with torch.no_grad(): eff_star = T_model.decode(f0, c0, zstar.view(len(b), T_model.n_cells, T_model.d), u0)
                terms["eff"] = F.mse_loss(eff_hat, eff_star); loss = loss + a.lam_e * terms["eff"]
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0); opt.step(); sched.step()
        for k, v in terms.items(): log.setdefault(k, []).append(float(v))
        if (it + 1) % 1000 == 0: print(f"[policy:{arm}] it {it+1} p={p:.2f} " + " ".join(f"{k}={np.mean(v[-1000:]):.4f}" for k, v in log.items()) + f" ({time.time()-t0:.0f}s)", flush=True)
    agent.eval(); return agent, chunk_mu, chunk_sd


@torch.no_grad()
def predict(agent, feats, sel, dev, chunk_mu, chunk_sd, z_override=None, bs=2048):
    out = []
    idx = np.flatnonzero(sel)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]; fq = feats["FQ"][b].to(dev).float(); task = torch.as_tensor(feats["task"][b], device=dev)
        z = None if z_override is None else torch.as_tensor(z_override[b], device=dev)
        pred, _ = agent(fq, task, z_for_head=z); out.append((pred * chunk_sd + chunk_mu).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=["bc", "ours", "ours_noeff", "univla"])
    ap.add_argument("--tokenizer", default="s1_acdc_1ep"); ap.add_argument("--transport", default="s3_latp_1ep"); ap.add_argument("--univla-arm", default="s2mix_1ep")
    ap.add_argument("--iters", type=int, default=8000); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lam-z", type=float, default=1.0); ap.add_argument("--lam-e", type=float, default=1.0)
    ap.add_argument("--rollout", type=int, default=0, help="closed-loop starts per task (0 = offline only)"); ap.add_argument("--rollout-arms", nargs="*", default=["bc", "ours", "univla"]); ap.add_argument("--max-steps", type=int, default=220); ap.add_argument("--exec", type=int, default=S)
    ap.add_argument("--out", default="libero_policy")
    a = ap.parse_args(); dev = "cuda"; t0 = time.time()
    dino = Fz.load_dino(dev); G, idx = L.frame_features(dino, dev); T_, E_ = L.steps_of(idx)
    is_val = idx["is_val"].astype(bool); tr = ~is_val[E_]; te = is_val[E_]; task = idx["task_id"][E_].astype(np.int64)
    A = idx["actions"]; chunk = np.stack([A[t:t + K * S].reshape(-1) for t in T_]).astype(np.float32)
    H_idx = hist_index(T_, E_, idx); FQ = Fz.quadrants(G[torch.from_numpy(H_idx.reshape(-1))].float()).view(len(T_), HIST, 4, 768).half()
    F0 = G[torch.from_numpy(T_)]
    feats = dict(FQ=FQ, F0=F0, task=task, chunk=chunk, tr=tr, te=te, Zstar=None)
    print(f"[policy] steps train {tr.sum()} test {te.sum()} ({time.time()-t0:.0f}s)", flush=True)
    R = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "iters": a.iters}; csd = chunk[tr].std(0) + 1e-6
    def metrics(P): return dict(l1=float((np.abs(P - chunk[te]) / csd).mean()), tok_acc=float((L.token_bins(chunk[tr], P) == L.token_bins(chunk[tr], chunk[te])).mean()))
    agents = {}
    for arm in a.arms:
        T_model = None; feats_arm = dict(feats)
        if arm in ("ours", "ours_noeff"):
            p = os.path.join(CKPT, f"{a.tokenizer}.pt"); tok, dstats, _ = load_ckpt(p if os.path.exists(p) else legacy_path(a.tokenizer), dev); tok.eval()
            feats_arm["Zstar"] = L.our_latents(tok, dstats, G, T_, dev).astype(np.float32)
            if arm == "ours":
                pT = os.path.join(CKPT, f"{a.transport}.pt"); T_model, _, _ = load_ckpt(pT if os.path.exists(pT) else legacy_path(a.transport), dev); T_model.eval(); T_model.requires_grad_(False)
            lam_e = a.lam_e if arm == "ours" else 0.0
        elif arm == "univla":
            Z, _ = L.univla_latents(dev, T_, L.make_frame_reader(idx, dev), arm=a.univla_arm); feats_arm["Zstar"] = Z.astype(np.float32); lam_e = 0.0
        else: lam_e = 0.0
        a_arm = argparse.Namespace(**{**vars(a), "lam_e": lam_e})
        agent, mu, sd = train_agent(arm, feats_arm, dev, a_arm, T_model)
        r = dict(chain=metrics(predict(agent, feats_arm, te, dev, mu, sd)))
        if feats_arm["Zstar"] is not None:
            r["oracle"] = metrics(predict(agent, feats_arm, te, dev, mu, sd, z_override=feats_arm["Zstar"]))
            with torch.no_grad():
                zh = torch.cat([agent.pi(feats_arm["FQ"][b].to(dev).float(), torch.as_tensor(task[b], device=dev)).cpu() for b in np.array_split(np.flatnonzero(te), 20)]).numpy()
            zs = feats_arm["Zstar"][te]; r["latent_r2"] = float(1 - ((zh - zs) ** 2).sum() / ((zs - feats_arm["Zstar"][tr].mean(0)) ** 2).sum())
        R[arm] = r; agents[arm] = (agent, mu, sd, feats_arm["Zstar"]); print(f"[policy] {arm}: {r}", flush=True)
        json.dump(R, open(os.path.join(RESULTS, f"{a.out}.json"), "w"), indent=1)
    print("%-12s %10s %10s %10s %10s" % ("arm", "l1 chain", "tok chain", "l1 oracle", "latent_r2"))
    for arm, r in R.items():
        if isinstance(r, dict) and "chain" in r: print("%-12s %10.3f %10.3f %10s %10s" % (arm, r["chain"]["l1"], r["chain"]["tok_acc"], f"{r['oracle']['l1']:.3f}" if "oracle" in r else "-", f"{r['latent_r2']:.3f}" if "latent_r2" in r else "-"))
    if a.rollout: rollout({k: v for k, v in agents.items() if k in a.rollout_arms}, idx, a, dev, dino, R)
    print(f"[policy-done] {time.time()-t0:.0f}s", flush=True)


def rollout(agents, idx, a, dev, dino, R):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from .libero_closedloop import orientation
    b = benchmark.get_benchmark_dict()["libero_spatial"](); eps = libero.list_episodes(); is_val = idx["is_val"].astype(bool)
    R["rollout"] = {"n_init": a.rollout, "max_steps": a.max_steps, "per_task": {}}; flip = None; t0 = time.time()
    for ti in range(N_TASKS):
        tk = b.get_task(ti); init_states = b.get_task_init_states(ti)
        val_demos = [int(str(idx["demo"][e]).split("_")[1]) for e in range(len(eps)) if int(idx["task_id"][e]) == ti and is_val[e]][: a.rollout]
        env = OffScreenRenderEnv(bddl_file_name=f"{get_libero_path('bddl_files')}/{tk.problem_folder}/{tk.bddl_file}", camera_heights=128, camera_widths=128); env.seed(0)
        for name, (agent, mu, sd, _) in agents.items():
            succ = []
            for d in val_demos:
                env.reset(); obs = env.set_init_state(init_states[d])
                for _ in range(5): obs, _, _, _ = env.step(np.zeros(7))
                if flip is None:
                    e0 = next(e for e in range(len(eps)) if int(idx["task_id"][e]) == ti and int(str(idx["demo"][e]).split("_")[1]) == d)
                    flip = orientation(obs["agentview_image"], libero.load_episode_frames(eps[e0])[0])
                hist = []; done, steps = False, 0
                while not done and steps < a.max_steps:
                    img = torch.from_numpy(np.ascontiguousarray(flip(obs["agentview_image"]))).to(dev).permute(2, 0, 1)[None].float().div_(255.)
                    img = F.interpolate(img, size=(IMG, IMG), mode="bilinear", align_corners=False)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True): fq = Fz.quadrants(Fz.dino_g4(dino, img).float())[0]     # (4,768)
                    hist.append(fq); hist = hist[-HIST:]
                    fqh = torch.stack([hist[0]] * (HIST - len(hist)) + hist)[None]                                                       # (1,HIST,4,768)
                    with torch.no_grad(): pred, _ = agent(fqh, torch.tensor([ti], device=dev))
                    ch = ((pred * sd + mu).cpu().numpy()).reshape(K * S, 7)
                    for k in range(a.exec):
                        act = ch[k].copy(); act[6] = 1.0 if act[6] > 0 else -1.0
                        obs, _, done, _ = env.step(act); steps += 1
                        if done or steps >= a.max_steps: break
                succ.append(bool(done))
            R["rollout"]["per_task"].setdefault(name, {})[tk.language] = float(np.mean(succ))
            print(f"[policy-rollout] task {ti} {name}: {np.mean(succ):.2f} ({len(succ)} eps, {time.time()-t0:.0f}s)", flush=True)
        env.close()
        R["rollout"]["success"] = {n: float(np.mean(list(v.values()))) for n, v in R["rollout"]["per_task"].items()}
        json.dump(R, open(os.path.join(RESULTS, f"{a.out}.json"), "w"), indent=1)
    n_eff = min(a.rollout, 10)   # LIBERO-spatial keeps 10 held-out demos per task in our split; the count above is what was actually run
    print("[policy-rollout] SUCCESS RATE (libero_spatial, %d tasks x %d held-out inits per task):" % (N_TASKS, n_eff))
    for n, v in R["rollout"]["success"].items(): print("   %-14s %.3f" % (n, v))


if __name__ == "__main__":
    main()
