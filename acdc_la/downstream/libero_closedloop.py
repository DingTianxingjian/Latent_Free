"""Closed-loop SUCCESS RATE on LIBERO-spatial (the field's metric), for a latent-action policy built on a tokenizer.

Policy = the two heads of libero_offline trained on the 80% train demos:  pi(z | g4(o_t), task) -> g(a_{t:t+16} | z, g4(o_t)).
At test time, on the 20% held-out initial states of every task (the demo split, so the policy has never seen these
starts): render 128x128 agentview, DINO g4, predict z, decode the 16-step chunk, execute `--exec` steps, re-plan.
Success = the environment's task predicate within 300 steps (OpenVLA's LIBERO protocol uses 220 + slack).
Arms: --tags tokenizers (ours; K=1 tokenizers applied per segment), `univla` (official stage-2 codes: the policy predicts
the 16 VQ indices, looked up in the codebook), `nolatent` (g4 + task -> chunk directly; the floor).
Runs in the `libero` venv: PYTHONPATH=latent_free:envs/LIBERO MUJOCO_GL=egl.
    python -m acdc_la.downstream.libero_closedloop --tags s1_acdc_1ep s1_mean_1ep --arms univla nolatent --n-init 10
"""
import os, sys, json, time, argparse
import numpy as np
import torch, torch.nn.functional as F

from ..paths import RESULTS, CKPT, IMG
from ..common import features as Fz
from ..common.model import load_ckpt, legacy_path
from . import libero_offline as L
from .libero_offline import libero, K, S, N_TASKS


class ActE:
    """+E arm: policy -> z^, then the transport decoder's OUTPUT effect dF^ on the current frame -> action head."""
    def __init__(self, pol, decE, model, dev): self.pol, self.decE, self.model, self.dev, self.f0 = pol, decE, model, dev, None
    def __call__(self, x):
        return self.decE(np.concatenate([L.effect_feats(self.model, self.f0, np.arange(len(x)), self.pol(x), self.dev), x], 1))


class ActT:
    """+T arm: policy -> z^, then the transport decoder's hidden state on the CURRENT frame's 16 g4 tokens (set in .f0 by
    the rollout loop before each call) -> action head.  A class, not a closure: closures over a loop variable bind late."""
    def __init__(self, pol, decT, model, dev): self.pol, self.decT, self.model, self.dev, self.f0 = pol, decT, model, dev, None
    def __call__(self, x):
        return self.decT(np.concatenate([L.decoder_feats(self.model, self.f0, np.arange(len(x)), self.pol(x), self.dev), x], 1))


def orientation(env_img, hdf_img):
    """Which flip maps an env render onto the hdf5 frames the heads were trained on (data.py flips hdf5 rows)."""
    cands = {"none": lambda x: x, "vflip": lambda x: x[::-1], "rot180": lambda x: x[::-1, ::-1], "hflip": lambda x: x[:, ::-1]}
    sc = {k: float(np.corrcoef(f(env_img).astype(np.float32).ravel(), hdf_img.astype(np.float32).ravel())[0, 1]) for k, f in cands.items()}
    best = max(sc, key=sc.get); print(f"[closedloop] orientation {sc} -> {best}", flush=True)
    return cands[best]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="*", default=["s1_acdc_1ep"]); ap.add_argument("--arms", nargs="*", default=["univla", "nolatent", "T"], help="extra arms: univla, nolatent, T (= +T decoder-reuse variant of every --tags model)")
    ap.add_argument("--univla-arm", default="official_stage2"); ap.add_argument("--n-init", type=int, default=10, help="held-out initial states per task (of the 10 val demos)"); ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--exec", type=int, default=S, help="actions executed per re-plan"); ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--out", default="libero_closedloop")
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    dino = Fz.load_dino(dev); G, idx = L.frame_features(dino, dev); T_, E_ = L.steps_of(idx)
    is_val = idx["is_val"].astype(bool); tr = ~is_val[E_]; task = idx["task_id"][E_]
    A = idx["actions"]; chunk = np.stack([A[t:t + K * S].reshape(-1) for t in T_])
    obs_feat = Fz.quadrants(G[torch.from_numpy(T_)].float()).flatten(1).numpy(); onehot = np.eye(N_TASKS, dtype=np.float32)[task]
    X_obs = np.concatenate([obs_feat, onehot], 1)
    # ---- heads per arm: policy(X_obs) -> chunk (nolatent) or -> latent, and decoder([latent, X_obs]) -> chunk
    arms = {}
    for tag in a.tags:
        p = os.path.join(CKPT, f"{tag}.pt"); p = p if os.path.exists(p) else legacy_path(tag)
        model, dstats, _ = load_ckpt(p, dev); model.eval(); Z = L.our_latents(model, dstats, G, T_, dev)
        pol = L.fit_net(X_obs[tr], Z[tr], dev, a.steps); dec = L.fit_net(np.concatenate([Z[tr], X_obs[tr]], 1), chunk[tr], dev, a.steps)
        arms[tag] = lambda x, pol=pol, dec=dec: dec(np.concatenate([pol(x), x], 1))
        if model.K == K and "T" in a.arms:                          # +T: action head on the transport decoder's hidden state (u = prior mean, c = 0)
            Hz = L.decoder_feats(model, G, T_, Z, dev); decT = L.fit_net(np.concatenate([Hz[tr], X_obs[tr]], 1), chunk[tr], dev, a.steps)
            arms[tag + "+T"] = ActT(pol, decT, model, dev)
        if model.K == K and "E" in a.arms:
            Ez = L.effect_feats(model, G, T_, Z, dev); decE = L.fit_net(np.concatenate([Ez[tr], X_obs[tr]], 1), chunk[tr], dev, a.steps)
            arms[tag + "+E"] = ActE(pol, decE, model, dev)
    if "univla" in a.arms:
        Z, I = L.univla_latents(dev, T_, L.make_frame_reader(idx, dev), arm=a.univla_arm); import eval_rala2 as E
        cb = E.load_arm(a.univla_arm, dev)[0].vq_action.codebook.weight.detach().cpu().numpy()          # (16,128)
        pol = L.fit_net(X_obs[tr], I[tr], dev, a.steps, ce_classes=16); dec = L.fit_net(np.concatenate([Z[tr], X_obs[tr]], 1), chunk[tr], dev, a.steps)
        arms[f"univla_{a.univla_arm}"] = lambda x, pol=pol, dec=dec: dec(np.concatenate([cb[pol(x)].reshape(len(x), -1), x], 1))
    if "nolatent" in a.arms:
        pol0 = L.fit_net(X_obs[tr], chunk[tr], dev, a.steps); arms["nolatent"] = pol0
    print(f"[closedloop] heads trained for {list(arms)} ({time.time()-t0:.0f}s)", flush=True)
    # ---- rollouts
    b = benchmark.get_benchmark_dict()["libero_spatial"](); eps = libero.list_episodes()
    R = {"n_init": a.n_init, "max_steps": a.max_steps, "exec": a.exec, "per_task": {}}
    flip = None
    for ti in range(N_TASKS):
        tk = b.get_task(ti); init_states = b.get_task_init_states(ti)
        # held-out demos of this task = the val split's demo indices (demo_k <-> init state k)
        val_demos = [int(str(idx["demo"][e]).split("_")[1]) for e in range(len(eps)) if int(idx["task_id"][e]) == ti and is_val[e]][: a.n_init]
        env = OffScreenRenderEnv(bddl_file_name=f"{get_libero_path('bddl_files')}/{tk.problem_folder}/{tk.bddl_file}", camera_heights=128, camera_widths=128)
        env.seed(0); onehot_t = np.eye(N_TASKS, dtype=np.float32)[ti][None]
        for name, act_fn in arms.items():
            succ = []
            for d in val_demos:
                env.reset(); obs = env.set_init_state(init_states[d])
                for _ in range(5): obs, _, _, _ = env.step(np.zeros(7))                                  # settle, as OpenVLA does
                if flip is None:
                    e0 = next(e for e in range(len(eps)) if int(idx["task_id"][e]) == ti and int(str(idx["demo"][e]).split("_")[1]) == d)
                    flip = orientation(obs["agentview_image"], libero.load_episode_frames(eps[e0])[0])
                done, steps = False, 0
                while not done and steps < a.max_steps:
                    img = torch.from_numpy(np.ascontiguousarray(flip(obs["agentview_image"]))).to(dev).permute(2, 0, 1)[None].float().div_(255.)
                    img = F.interpolate(img, size=(IMG, IMG), mode="bilinear", align_corners=False)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                        g4 = Fz.dino_g4(dino, img).float()
                    x = np.concatenate([Fz.quadrants(g4).flatten(1).cpu().numpy(), onehot_t], 1)
                    if isinstance(act_fn, (ActT, ActE)): act_fn.f0 = g4.half().cpu()                                    # (1,16,768) for decoder_feats
                    ch = act_fn(x).reshape(K * S, 7)
                    for k in range(a.exec):
                        act = ch[k].copy(); act[6] = 1.0 if act[6] > 0 else -1.0
                        obs, _, done, _ = env.step(act); steps += 1
                        if done or steps >= a.max_steps: break
                succ.append(bool(done))
            R["per_task"].setdefault(name, {})[tk.language] = float(np.mean(succ))
            print(f"[closedloop] task {ti} {name}: {np.mean(succ):.2f} ({len(succ)} eps, {time.time()-t0:.0f}s)", flush=True)
        env.close()
        R["success"] = {n: float(np.mean(list(v.values()))) for n, v in R["per_task"].items()}
        json.dump(R, open(os.path.join(RESULTS, f"{a.out}.json"), "w"), indent=1)
    print("[closedloop] SUCCESS RATE (libero_spatial, %d tasks x %d held-out inits):" % (N_TASKS, a.n_init))
    for n, v in R["success"].items(): print("   %-20s %.3f" % (n, v))
    print(f"[closedloop-done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
